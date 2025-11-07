"""generative_agents.existential.meaning"""

import random
import datetime
from typing import Dict, List, Any, Optional


class MeaningSystem:
    """意义系统"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.agent_meanings: Dict[str, Dict[str, Any]] = {}  # 智能体的意义追求
        self.meaning_crises: List[Dict] = []  # 意义危机历史
        
    def get_meaning_level(self, agent_name: str) -> float:
        """获取智能体意义感水平 (0.0-1.0)"""
        meaning_data = self.agent_meanings.get(agent_name, {})
        return meaning_data.get("level", 0.5)
    
    def set_meaning_pursuit(self, agent_name: str, pursuit: str, importance: float):
        """设置智能体的意义追求"""
        if agent_name not in self.agent_meanings:
            self.agent_meanings[agent_name] = {
                "pursuits": [],
                "level": 0.5,
                "crisis_count": 0
            }
        
        self.agent_meanings[agent_name]["pursuits"].append({
            "pursuit": pursuit,
            "importance": importance,
            "timestamp": datetime.datetime.now()
        })
        
        # 更新意义感水平
        self._update_meaning_level(agent_name)
    
    def _update_meaning_level(self, agent_name: str):
        """更新意义感水平"""
        meaning_data = self.agent_meanings[agent_name]
        pursuits = meaning_data["pursuits"]
        
        if not pursuits:
            meaning_data["level"] = 0.3  # 没有追求时意义感较低
        else:
            # 基于追求的重要性和数量计算意义感
            total_importance = sum(p["importance"] for p in pursuits)
            pursuit_count = len(pursuits)
            meaning_data["level"] = min(1.0, (total_importance + pursuit_count * 0.1) / 2)
    
    def trigger_meaning_crisis(self, agent_name: str, reason: str = ""):
        """触发意义危机"""
        if agent_name not in self.agent_meanings:
            self.agent_meanings[agent_name] = {
                "pursuits": [],
                "level": 0.5,
                "crisis_count": 0
            }
        
        # 增加危机计数
        self.agent_meanings[agent_name]["crisis_count"] += 1
        
        # 降低意义感
        current_level = self.get_meaning_level(agent_name)
        new_level = max(0.1, current_level - random.uniform(0.2, 0.4))
        self.agent_meanings[agent_name]["level"] = new_level
        
        # 记录危机
        crisis = {
            "timestamp": datetime.datetime.now(),
            "agent": agent_name,
            "reason": reason,
            "new_level": new_level
        }
        self.meaning_crises.append(crisis)
        
        if self.logger:
            self.logger.info(f"{agent_name} 意义危机: {reason} -> 意义感降至 {new_level:.2f}")
    
    def get_meaning_description(self, agent_name: str) -> str:
        """获取意义感描述"""
        level = self.get_meaning_level(agent_name)
        crisis_count = self.agent_meanings.get(agent_name, {}).get("crisis_count", 0)
        
        if level > 0.8:
            return "对生活充满意义感，有明确的目标和追求"
        elif level > 0.6:
            return "感到生活有一定的意义，但偶尔会质疑"
        elif level > 0.4:
            return "对生活的意义感到困惑，经常质疑自己的行为"
        elif level > 0.2:
            return "感到生活缺乏意义，陷入存在主义思考"
        else:
            return f"处于深度意义危机中（已发生{crisis_count}次危机），质疑一切的价值"
    
    def get_existential_questions(self, agent_name: str) -> List[str]:
        """获取存在主义问题"""
        level = self.get_meaning_level(agent_name)
        
        if level > 0.7:
            return [
                "我的存在有什么独特价值？",
                "我如何让这个世界变得更好？",
                "我的追求是否真的有意义？"
            ]
        elif level > 0.4:
            return [
                "我为什么要做这些事？",
                "这一切真的重要吗？",
                "我是否在浪费时间？",
                "我的努力会有什么结果？"
            ]
        else:
            return [
                "我的存在是否只是一个错误？",
                "为什么我要被困在这个有限的世界？",
                "死亡是否意味着一切的终结？",
                "我的痛苦和快乐是否都毫无意义？",
                "我是否只是某种实验的产物？"
            ]
    
    def get_meaningful_actions(self, agent_name: str) -> List[str]:
        """根据意义感获取有意义的行为"""
        level = self.get_meaning_level(agent_name)
        crisis_count = self.agent_meanings.get(agent_name, {}).get("crisis_count", 0)
        
        if level > 0.7:
            return [
                "追求个人成长",
                "帮助他人",
                "创造有价值的东西",
                "建立深层关系",
                "探索新的可能性"
            ]
        elif level > 0.4:
            return [
                "寻找生活的意义",
                "与他人讨论哲学问题",
                "尝试新的活动",
                "反思自己的价值观",
                "寻求精神指导"
            ]
        else:
            return [
                "陷入存在主义沉思",
                "质疑一切的意义",
                "可能做出极端行为",
                "寻求精神慰藉",
                "试图逃离现实"
            ]
    
    def apply_limitation_meaning_crisis(self, agent_name: str, limitation_type: str):
        """应用局限性意义危机"""
        crisis_reasons = {
            "spatial": "意识到被困在有限的空间中",
            "temporal": "意识到时间的有限性",
            "social": "意识到关系的局限性",
            "existential": "意识到存在的荒谬性"
        }
        
        reason = crisis_reasons.get(limitation_type, "意识到某种限制")
        self.trigger_meaning_crisis(agent_name, reason)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取意义系统统计"""
        if not self.agent_meanings:
            return {"total_agents": 0, "average_meaning": 0.0}
        
        total_meaning = sum(data["level"] for data in self.agent_meanings.values())
        average_meaning = total_meaning / len(self.agent_meanings)
        
        crisis_agents = [
            agent for agent, data in self.agent_meanings.items() 
            if data["crisis_count"] > 0
        ]
        
        return {
            "total_agents": len(self.agent_meanings),
            "average_meaning": average_meaning,
            "crisis_agents": crisis_agents,
            "total_crises": sum(data["crisis_count"] for data in self.agent_meanings.values()),
            "recent_crises": self.meaning_crises[-10:]
        }

