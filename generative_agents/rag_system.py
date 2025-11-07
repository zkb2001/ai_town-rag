import json
import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# 禁用HuggingFace符号链接警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class ChineseEmbeddings(Embeddings):
    """基于Ollama的中文嵌入模型包装器"""
    
    def __init__(self, model_name: str = "bge-m3:latest", base_url: str = "http://localhost:11434"):
        """
        初始化中文嵌入模型
        使用Ollama的bge-m3模型，对中文支持很好
        """
        self.model = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
        print(f"已加载Ollama中文嵌入模型: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.model.embed_query(text)

class JSONRAGSystem:
    """基于JSON文件的RAG问答系统"""
    
    def __init__(self, result_dir: str = "results/compressed", cache_dir: str = "vector_cache"):
        # 规范化结果目录为相对于当前文件的路径，避免工作目录变化导致找不到项目
        base_dir = Path(__file__).parent
        provided_path = Path(result_dir)
        self.result_dir = (provided_path if provided_path.is_absolute() else (base_dir / provided_path)).resolve()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # 调试输出：显示解析后的项目根目录
        print(f"项目根目录: {self.result_dir}")
        
        # 初始化嵌入模型
        self.embedder = ChineseEmbeddings()
        self.embedder_model_name = "bge-m3:latest"
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        )
        
        # 初始化LLM
        self.llm = self._init_llm()
        
        # 向量存储
        self.vectorstore = None
        self.conversation_store = None
        
        # 加载环境变量
        load_dotenv()
    
    def _init_llm(self):
        """初始化LLM"""
        try:
            return OllamaLLM(
                model="qwen3:4b-q4_K_M",
                base_url="http://localhost:11434",
                temperature=0.7
            )
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            print("请确保Ollama服务正在运行，并且已下载qwen3:4b-q4_K_M模型")
            return None
    
    def _get_json_hash(self, file_path: str) -> str:
        """生成JSON文件的唯一标识符"""
        content = f"{file_path}_{self.embedder_model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_project_hash(self, project_path: str) -> str:
        """生成项目的唯一标识符"""
        content = f"{project_path}_{self.embedder_model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_vectorized(self, project_path: str) -> bool:
        """检查项目是否已向量化（兼容不同哈希/路径生成方式）"""
        project_dir = Path(project_path)
        if not project_dir.exists():
            return False

        try:
            # 直接通过模式匹配检测是否存在向量目录和对应的metadata文件
            vector_dirs = list(project_dir.glob("*_vector"))
            metadata_files = list(project_dir.glob("*_metadata.pkl"))

            if not vector_dirs or not metadata_files:
                return False

            # 要求有同前缀的一对（更稳健）
            vector_prefixes = {p.name[:-len("_vector")] for p in vector_dirs if p.is_dir()}
            metadata_prefixes = {p.name[:-len("_metadata.pkl")] for p in metadata_files}

            return len(vector_prefixes.intersection(metadata_prefixes)) > 0
        except Exception:
            return False
    
    def _save_vectorized(self, file_path: str, vectorstore, documents: List[Document], metadata: Dict):
        """保存向量化数据"""
        file_hash = self._get_json_hash(file_path)
        
        # 保存向量存储
        vector_file = self.cache_dir / f"{file_hash}_vector.pkl"
        with open(vector_file, 'wb') as f:
            pickle.dump(vectorstore, f)
        
        # 保存文档和元数据
        metadata_file = self.cache_dir / f"{file_hash}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'documents': documents,
                'metadata': metadata
            }, f)
        
        print(f"✅ 已保存向量化数据: {file_path}")
    
    def _save_project_vectorized(self, project_dir: Path, vectorstore, documents: List[Document], metadata: Dict):
        """保存项目向量化数据到项目目录"""
        project_hash = self._get_project_hash(str(project_dir))
        
        # 在项目目录下创建向量化文件
        vector_file = project_dir / f"{project_hash}_vector.pkl"
        metadata_file = project_dir / f"{project_hash}_metadata.pkl"
        
        try:
            # 保存向量存储 - 使用FAISS的save方法
            vectorstore.save_local(str(vector_file.with_suffix('')))
            
            # 保存文档和元数据
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': documents,
                    'metadata': metadata
                }, f)
            
            print(f"✅ 已保存项目向量化数据: {project_dir.name}")
        except Exception as e:
            print(f"❌ 保存向量化数据失败: {e}")
            # 如果FAISS保存失败，尝试保存文档和元数据
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': documents,
                    'metadata': metadata
                }, f)
    
    def _load_vectorized(self, file_path: str) -> Tuple[FAISS, List[Document], Dict]:
        """加载向量化数据"""
        file_hash = self._get_json_hash(file_path)
        
        vector_file = self.cache_dir / f"{file_hash}_vector.pkl"
        metadata_file = self.cache_dir / f"{file_hash}_metadata.pkl"
        
        with open(vector_file, 'rb') as f:
            vectorstore = pickle.load(f)
        
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            documents = data['documents']
            metadata = data['metadata']
        
        print(f"✅ 已从缓存加载: {file_path}")
        return vectorstore, documents, metadata
    
    def _extract_movement_json(self, json_file: Path) -> List[Document]:
        """从movement.json文件中提取内容并转换为文档"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # 处理movement.json的特定结构
            if isinstance(data, list):
                # 处理移动记录列表
                for i, movement in enumerate(data):
                    if isinstance(movement, dict):
                        content = f"移动记录 {i+1}:\n"
                        content += f"时间: {movement.get('timestamp', '未知')}\n"
                        content += f"角色: {movement.get('character', '未知')}\n"
                        content += f"位置: {movement.get('location', '未知')}\n"
                        content += f"动作: {movement.get('action', '未知')}\n"
                        if 'details' in movement:
                            content += f"详情: {json.dumps(movement['details'], ensure_ascii=False, indent=2)}"
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': str(json_file),
                                'type': 'movement',
                                'index': i,
                                'file_name': json_file.name,
                                'file_path': str(json_file)
                            }
                        )
                        documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ 处理movement.json文件失败 {json_file}: {e}")
            return []
    
    def _extract_simulation_md(self, md_file: Path) -> List[Document]:
        """从simulation.md文件中提取内容并转换为文档"""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按段落分割markdown内容
            paragraphs = content.split('\n\n')
            documents = []
            
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    doc = Document(
                        page_content=paragraph.strip(),
                        metadata={
                            'source': str(md_file),
                            'type': 'simulation',
                            'paragraph': i,
                            'file_name': md_file.name,
                            'file_path': str(md_file)
                        }
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ 处理simulation.md文件失败 {md_file}: {e}")
            return []
    
    def _process_project_files(self, project_dir: Path) -> bool:
        """处理项目文件夹中的movement.json和simulation.md"""
        try:
            movement_file = project_dir / "movement.json"
            simulation_file = project_dir / "simulation.md"
            
            # 检查文件是否存在
            if not movement_file.exists() and not simulation_file.exists():
                print(f"项目文件夹中未找到movement.json或simulation.md: {project_dir.name}")
                return False
            
            # 检查是否已向量化
            project_hash = self._get_project_hash(str(project_dir))
            if self._check_vectorized(project_hash):
                print(f"跳过已处理的项目: {project_dir.name}")
                return True
            
            print(f"处理项目: {project_dir.name}")
            
            # 合并处理movement.json和simulation.md
            combined_content = ""
            file_info = []
            
            # 处理movement.json
            if movement_file.exists():
                print(f"处理movement.json...")
                with open(movement_file, 'r', encoding='utf-8') as f:
                    movement_data = json.load(f)
                
                # 提取关键信息
                movement_text = f"=== MOVEMENT DATA ===\n"
                movement_text += f"开始时间: {movement_data.get('start_datetime', '未知')}\n"
                movement_text += f"步长: {movement_data.get('stride', '未知')}\n"
                movement_text += f"每秒步数: {movement_data.get('sec_per_step', '未知')}\n\n"
                
                # 添加初始位置信息
                if 'persona_init_pos' in movement_data:
                    movement_text += "初始位置:\n"
                    for persona, pos in movement_data['persona_init_pos'].items():
                        movement_text += f"  {persona}: {pos}\n"
                
                # 添加移动轨迹数据（采样显示）
                if 'trajectory' in movement_data:
                    trajectory = movement_data['trajectory']
                    movement_text += f"\n移动轨迹数据 (共{len(trajectory)}条记录):\n"
                    # 只显示前10条和最后10条记录
                    for i, record in enumerate(trajectory[:10]):
                        movement_text += f"  {i+1}: {record}\n"
                    if len(trajectory) > 20:
                        movement_text += f"  ... (省略{len(trajectory)-20}条记录) ...\n"
                    for i, record in enumerate(trajectory[-10:], len(trajectory)-9):
                        movement_text += f"  {i}: {record}\n"
                
                combined_content += movement_text + "\n\n"
                file_info.append("movement.json")
                print(f"movement.json: 已提取关键信息")
            
            # 处理simulation.md
            if simulation_file.exists():
                print(f"处理simulation.md...")
                with open(simulation_file, 'r', encoding='utf-8') as f:
                    simulation_content = f.read()
                
                combined_content += f"=== SIMULATION DATA ===\n{simulation_content}\n\n"
                file_info.append("simulation.md")
                print(f"simulation.md: 已读取内容")
            
            if not combined_content.strip():
                print(f"项目内容为空: {project_dir.name}")
                return False
            
            # 创建单个文档
            doc = Document(
                page_content=combined_content,
                metadata={
                    'source': str(project_dir),
                    'project_name': project_dir.name,
                    'files': file_info,
                    'file_name': f"{project_dir.name}_combined",
                    'file_path': str(project_dir)
                }
            )
            
            # 分割文档
            chunks = self.text_splitter.split_documents([doc])
            
            if not chunks:
                print(f"分割后无有效内容: {project_dir.name}")
                return False
            
            # 创建向量存储
            vectorstore = FAISS.from_documents(chunks, self.embedder)
            
            # 保存元数据
            metadata = {
                'project_name': project_dir.name,
                'project_path': str(project_dir),
                'movement_file': str(movement_file) if movement_file.exists() else None,
                'simulation_file': str(simulation_file) if simulation_file.exists() else None,
                'total_chunks': len(chunks),
                'processed_time': time.time()
            }
            
            # 保存向量化数据到项目目录
            self._save_project_vectorized(project_dir, vectorstore, chunks, metadata)
            
            print(f"项目处理完成: {project_dir.name} ({len(chunks)} 个块)")
            return True
            
        except Exception as e:
            print(f"处理项目失败 {project_dir.name}: {e}")
            return False
    
    def extract_and_vectorize(self, project_name: str) -> bool:
        """处理指定项目的movement.json和simulation.md文件"""
        project_dir = self.result_dir / project_name
        
        if not project_dir.exists():
            print(f"❌ 项目目录不存在: {project_dir}")
            return False
        
        print(f"🔍 处理项目: {project_dir}")
        
        # 检查必要文件
        movement_file = project_dir / "movement.json"
        simulation_file = project_dir / "simulation.md"
        
        if not movement_file.exists() and not simulation_file.exists():
            print(f"❌ 项目目录中未找到movement.json或simulation.md: {project_dir}")
            return False
        
        # 处理项目文件
        return self._process_project_files(project_dir)
    
    def _load_all_vectorized(self) -> FAISS:
        """加载所有已向量化的数据"""
        if not self.result_dir.exists():
            return self._create_empty_vectorstore()
        
        vectorstores = []
        metadata_list = []
        loaded_projects = []
        
        # 遍历所有项目目录，查找向量化文件
        for project_dir in self.result_dir.iterdir():
            if project_dir.is_dir():
                # 查找项目目录中的向量化文件
                for vector_dir in project_dir.glob("*_vector"):
                    if vector_dir.is_dir():
                        try:
                            # 使用FAISS的load_local方法，允许加载pickle文件
                            vectorstore = FAISS.load_local(str(vector_dir), self.embedder, allow_dangerous_deserialization=True)
                            vectorstores.append(vectorstore)
                            loaded_projects.append(project_dir.name)
                            
                            # 加载对应的元数据
                            metadata_file = project_dir / f"{vector_dir.stem}_metadata.pkl"
                            if metadata_file.exists():
                                with open(metadata_file, 'rb') as f:
                                    metadata = pickle.load(f)
                                    metadata_list.append(metadata)
                            
                        except Exception as e:
                            print(f"加载向量文件失败 {vector_dir}: {e}")
                            continue
        
        if vectorstores:
            print(f"已加载 {len(loaded_projects)} 个项目的向量化数据: {', '.join(loaded_projects)}")
        
        if not vectorstores:
            return self._create_empty_vectorstore()
        
        # 合并所有向量存储
        if len(vectorstores) == 1:
            return vectorstores[0]
        else:
            combined_store = vectorstores[0]
            for store in vectorstores[1:]:
                combined_store.merge_from(store)
            return combined_store
    
    def _create_empty_vectorstore(self) -> FAISS:
        """创建空的向量存储"""
        embed_dims = len(self.embedder.embed_query("test"))
        return FAISS(
            embedding_function=self.embedder,
            index=IndexFlatL2(embed_dims),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            normalize_L2=False
        )
    
    #这里每次提问加载20个块
    def _retrieve_documents(self, query: str, k: int = 20) -> List[Document]:
        """检索相关文档"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return []
    
    def _format_context(self, docs: List[Document]) -> str:
        """格式化检索到的文档"""
        if not docs:
            return "未找到相关文档。"
        
        context = "相关文档内容：\n\n"
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('file_name', '未知文件')
            context += f"文档 {i} (来源: {source}):\n"
            context += f"{doc.page_content}\n\n"
        
        return context
    
    def query(self, question: str):
        """查询问答 - 流式输出"""
        if not self.llm:
            yield "❌ LLM未初始化，请确保Ollama服务正在运行"
            return
        
        # 检索相关文档
        docs = self._retrieve_documents(question)
        context = self._format_context(docs)
        
        # 构建提示
        prompt = f"""你是一个专业的文档分析助手。请基于以下文档内容回答用户的问题。

{context}

用户问题：{question}

请按照以下要求回答：
1. 基于文档的实际内容进行回答，不要编造信息
2. 如果文档内容不足以回答问题，请明确说明
3. 回答要准确、有条理
4. 如果涉及多个文档，请指出信息来源

请开始回答："""
        
        try:
            # 使用流式输出
            for chunk in self.llm.stream(prompt):
                yield chunk
        except Exception as e:
            yield f"❌ 生成回答失败: {str(e)}"
    
    def get_available_projects(self) -> List[str]:
        """获取可用的项目列表（包含movement.json或simulation.md的项目）"""
        if not self.result_dir.exists():
            return []
        
        projects = []
        for item in self.result_dir.iterdir():
            if item.is_dir():
                # 检查是否包含movement.json或simulation.md
                movement_file = item / "movement.json"
                simulation_file = item / "simulation.md"
                
                if movement_file.exists() or simulation_file.exists():
                    projects.append(item.name)
        
        return sorted(projects)
    
    def get_project_stats(self, project_name: str) -> Dict[str, Any]:
        """获取项目统计信息"""
        project_dir = self.result_dir / project_name
        if not project_dir.exists():
            return {"error": "项目不存在"}
        
        # 检查必要文件
        movement_file = project_dir / "movement.json"
        simulation_file = project_dir / "simulation.md"
        
        files_info = []
        if movement_file.exists():
            files_info.append("movement.json")
        if simulation_file.exists():
            files_info.append("simulation.md")
        
        # 检查是否已向量化
        is_vectorized = self._check_vectorized(str(project_dir))
        
        return {
            "total_files": len(files_info),
            "files": files_info,
            "vectorized": is_vectorized,
            "status": "已向量化" if is_vectorized else "未向量化"
        }
    
    def initialize_system(self):
        """初始化系统"""
        print("初始化RAG系统...")
        
        # 显示可用项目
        available_projects = self.get_available_projects()
        if available_projects:
            print(f"找到 {len(available_projects)} 个可用项目: {', '.join(available_projects)}")
        else:
            print("未找到任何项目")
        
        # 加载向量存储
        self.vectorstore = self._load_all_vectorized()
        
        # 初始化对话存储
        self.conversation_store = self._create_empty_vectorstore()
        
        print("RAG系统初始化完成")

def create_gradio_interface():
    """创建Gradio界面"""
    rag_system = JSONRAGSystem()
    rag_system.initialize_system()
    
    with gr.Blocks(
        title="JSON文档RAG问答系统",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            width: 100%;
            max-width: none;
            margin: 0;
            padding: 0;
        }
        .header-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            text-align: center;
            width: 100%;
        }
        .left-panel {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-right: 10px;
        }
        .right-panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        .project-stats {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .status-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        # 头部
        with gr.Column(elem_classes=["header-section"]):
            gr.HTML("""
                <h1 style="margin: 0; font-size: 2.5rem;">📚 JSON文档RAG问答系统</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                    基于JSON文档的智能问答系统，支持中文语义搜索
                </p>
            """)
        
        # 主界面 - 两列布局
        with gr.Row():
            # 左侧面板，scale的意思是这个列的宽度是总宽度的1/3
            with gr.Column(scale=1, elem_classes=["left-panel"]):
                # 项目管理
                gr.HTML('<h3>🎯 项目管理</h3>')
                
                # 项目选择
                project_dropdown = gr.Dropdown(
                    choices=rag_system.get_available_projects(),
                    label="选择项目",
                    interactive=True
                )
                
                # 项目统计
                stats_display = gr.HTML(
                    value="<p>请选择一个项目查看统计信息</p>",
                    label="项目统计"
                )
                
                # 处理按钮
                process_btn = gr.Button(
                    "🚀 开始向量化处理",
                    variant="primary",
                    size="lg"
                )
                
                # 处理状态
                process_status = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                    value="等待处理..."
                )
                
                # 刷新按钮
                refresh_btn = gr.Button("🔄 刷新项目列表", variant="secondary")
                
                # 系统状态
                gr.HTML('<h3>📊 系统状态</h3>')
                system_status = gr.HTML(
                    value="<p>系统已就绪</p>",
                    label="系统状态"
                )
            
            # 右侧面板
            with gr.Column(scale=2, elem_classes=["right-panel"]):
                # 智能问答
                gr.HTML('<h3>💬 智能问答</h3>')
                
                # 聊天记录
                chatbot = gr.Chatbot(
                    height=400,
                    show_label=False,
                    show_copy_button=True,
                    type="messages"
                )
                
                # 输入区域
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="请输入您的问题...",
                        show_label=False,
                        scale=4
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                
                # 清空按钮
                clear_btn = gr.Button("清空对话", variant="secondary")
        
        # 全局状态 - 使用None初始化，避免深拷贝问题
        current_vectorstore = gr.State(None)
        current_conversation_store = gr.State(None)
        
        def update_project_stats(project_name):
            """更新项目统计信息"""
            if not project_name:
                return "<p>请选择一个项目</p>"
            
            stats = rag_system.get_project_stats(project_name)
            if "error" in stats:
                return f"<p style='color: red;'>{stats['error']}</p>"
            
            return f"""
            <div class="project-stats">
                <p><strong>项目文件:</strong> {', '.join(stats['files'])}</p>
                <p><strong>向量化状态:</strong> <span style="color: {'green' if stats['vectorized'] else 'red'}">{stats['status']}</span></p>
                <p><strong>文件数量:</strong> {stats['total_files']}</p>
            </div>
            """
        
        def process_project(project_name, current_vectorstore_state):
            """处理项目向量化"""
            if not project_name:
                return "请先选择一个项目", current_vectorstore_state
            
            try:
                # 执行向量化处理
                success = rag_system.extract_and_vectorize(project_name)
                
                if success:
                    # 重新加载向量存储
                    rag_system.vectorstore = rag_system._load_all_vectorized()
                    return f"✅ 项目 {project_name} 处理完成！", "loaded"  # 返回状态标识而不是对象
                else:
                    return f"❌ 项目 {project_name} 处理失败", current_vectorstore_state
                    
            except Exception as e:
                return f"❌ 处理出错: {str(e)}", current_vectorstore_state
        
        def chat_with_rag(message, history, vectorstore_state):
            """RAG聊天功能 - 流式输出"""
            if not message.strip():
                return history, ""
            
            # 添加用户消息到历史记录（使用messages格式）
            history.append({"role": "user", "content": message})
            
            # 先显示用户消息
            yield history, ""
            
            # 获取AI回答（流式）
            try:
                # 直接使用rag_system的向量存储，不依赖状态
                response_generator = rag_system.query(message)
                
                # 初始化AI回答
                ai_response = ""
                
                # 流式更新回答
                for chunk in response_generator:
                    if chunk:
                        ai_response += chunk
                        # 添加AI回答到历史记录
                        if len(history) > 0 and history[-1]["role"] == "user":
                            # 如果最后一条是用户消息，添加AI回答
                            history.append({"role": "assistant", "content": ai_response})
                        else:
                            # 更新最后一条AI消息
                            history[-1] = {"role": "assistant", "content": ai_response}
                        yield history, ""
                
            except Exception as e:
                error_msg = f"❌ 查询失败: {str(e)}"
                history.append({"role": "assistant", "content": error_msg})
                yield history, ""
        
        def refresh_projects():
            """刷新项目列表"""
            projects = rag_system.get_available_projects()
            return gr.Dropdown(choices=projects)
        
        # 绑定事件
        project_dropdown.change(
            update_project_stats,
            inputs=[project_dropdown],
            outputs=[stats_display]
        )
        
        process_btn.click(
            process_project,
            inputs=[project_dropdown, current_vectorstore],
            outputs=[process_status, current_vectorstore]
        )
        
        send_btn.click(
            chat_with_rag,
            inputs=[msg_input, chatbot, current_vectorstore],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            chat_with_rag,
            inputs=[msg_input, chatbot, current_vectorstore],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(lambda: [], outputs=[chatbot])
        refresh_btn.click(refresh_projects, outputs=[project_dropdown])
    
    return demo

if __name__ == "__main__":
    # 启动界面
    demo = create_gradio_interface()
    demo.queue().launch(
        server_name='127.0.0.1',
        server_port=7861,
        share=False,
        debug=True
    )

