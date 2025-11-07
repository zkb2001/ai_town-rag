"""generative_agents.agent"""

import os
import math
import random
import datetime

from modules import memory, prompt, utils
from modules.model.llm_model import create_llm_model
from modules.memory.associate import Concept
from modules.existential import MeaningSystem


class Agent:
    def __init__(self, config, maze, conversation, logger):
        self.name = config["name"]
        self.maze = maze
        self.conversation = conversation
        self._llm = None
        self.logger = logger

        # agent config
        self.percept_config = config["percept"]
        self.think_config = config["think"]
        self.chat_iter = config["chat_iter"]

        # memory
        self.spatial = memory.Spatial(**config["spatial"])
        self.schedule = memory.Schedule(**config["schedule"])
        self.associate = memory.Associate(
            os.path.join(config["storage_root"], "associate"), **config["associate"]
        )
        self.concepts, self.chats = [], config.get("chats", [])

        # prompt
        self.scratch = prompt.Scratch(self.name, config["currently"], config["scratch"])

        # status
        status = {"poignancy": 0}
        self.status = utils.update_dict(status, config.get("status", {}))
        self.plan = config.get("plan", {})

        # record
        self.last_record = utils.get_timer().daily_duration()

        # 存在主义系统 - 只保留意义系统
        self.meaning_system = MeaningSystem(logger)
        
        # 初始化高强度的意义危机状态
        self.meaning_system.agent_meanings[self.name] = {
            "pursuits": [],
            "level": 0.2,  # 低意义感
            "crisis_count": 1  # 已经经历过危机
        }
        
        # 存在主义状态 - 只关注意义感
        self.existential_state = {
            "meaning_level": 0.2,      # 低意义感
            "crisis_count": 0
        }

        # action and events
        if "action" in config:
            self.action = memory.Action.from_dict(config["action"])
            tiles = self.maze.get_address_tiles(self.get_event().address)
            config["coord"] = random.choice(list(tiles))
        else:
            tile = self.maze.tile_at(config["coord"])
            address = tile.get_address("game_object", as_list=True)
            self.action = memory.Action(
                memory.Event(self.name, address=address),
                memory.Event(address[-1], address=address),
            )

        # update maze
        self.coord, self.path = None, None
        self.move(config["coord"], config.get("path"))
        if self.coord is None:
            self.coord = config["coord"]

    def abstract(self):
        des = {
            "name": self.name,
            "currently": self.scratch.currently,
            "tile": self.maze.tile_at(self.coord).abstract(),
            "status": self.status,
            "concepts": {c.node_id: c.abstract() for c in self.concepts},
            "chats": self.chats,
            "action": self.action.abstract(),
            "associate": self.associate.abstract(),
        }
        if self.schedule.scheduled():
            des["schedule"] = self.schedule.abstract()
        if self.llm_available():
            des["llm"] = self._llm.get_summary()
        # if self.plan.get("path"):
        #     des["path"] = "-".join(
        #         ["{},{}".format(c[0], c[1]) for c in self.plan["path"]]
        #     )
        return des

    def __str__(self):
        return utils.dump_dict(self.abstract())

    def reset(self):
        if not self._llm:
            self._llm = create_llm_model(self.think_config["llm"])

    def completion(self, func_hint, *args, **kwargs):
        assert hasattr(
            self.scratch, "prompt_" + func_hint
        ), "Can not find func prompt_{} from scratch".format(func_hint)
        func = getattr(self.scratch, "prompt_" + func_hint)
        
        # 添加存在主义上下文
        kwargs["existential_context"] = self._get_existential_context()
        
        prompt = func(*args, **kwargs)
        title, msg = "{}.{}".format(self.name, func_hint), {}
        if self.llm_available():
            self.logger.info("{} -> {}".format(self.name, func_hint))
            output = self._llm.completion(**prompt, caller=func_hint)
            responses = self._llm.meta_responses
            msg = {"<PROMPT>": "\n" + prompt["prompt"] + "\n"}
            msg.update(
                {
                    "<RESPONSE[{}/{}]>".format(idx+1, len(responses)): "\n" + r + "\n"
                    for idx, r in enumerate(responses)
                }
            )
        else:
            output = prompt.get("failsafe")
        msg["<OUTPUT>"] = "\n" + str(output) + "\n"
        self.logger.debug(utils.block_msg(title, msg))
        return output

    def think(self, status, agents, time_step=0):
        # 更新存在主义状态
        self._update_existential_state(agents, time_step)
        
        events = self.move(status["coord"], status.get("path"))
        plan, _ = self.make_schedule()

        # 根据存在主义状态调整计划
        plan = self._adjust_plan_for_existential_state(plan)
        
        # 存在主义反思 - 在关键决策点进行反思
        if self._should_existential_reflect():
            plan = self._existential_reflection(plan)

        if (plan["describe"] == "sleeping" or "睡" in plan["describe"]) and self.is_awake():
            self.logger.info("{} is going to sleep...".format(self.name))
            address = self.spatial.find_address("睡觉", as_list=True)
            tiles = self.maze.get_address_tiles(address)
            coord = random.choice(list(tiles))
            events = self.move(coord)
            self.action = memory.Action(
                memory.Event(self.name, "正在", "睡觉", address=address, emoji="😴"),
                memory.Event(
                    address[-1],
                    "被占用",
                    self.name,
                    address=address,
                    emoji="🛌",
                ),
                duration=plan["duration"],
                start=utils.get_timer().daily_time(plan["start"]),
            )
        if self.is_awake():
            self.percept()
            self.make_plan(agents)
            self.reflect()
        else:
            if self.action.finished():
                self.action = self._determine_action()

        emojis = {}
        if self.action:
            emojis[self.name] = {"emoji": self.get_event().emoji, "coord": self.coord}
        for eve, coord in events.items():
            if eve.subject in agents:
                continue
            emojis[":".join(eve.address)] = {"emoji": eve.emoji, "coord": coord}
        self.plan = {
            "name": self.name,
            "path": self.find_path(agents),
            "emojis": emojis,
        }
        return self.plan

    def _update_existential_state(self, agents, time_step):
        """更新存在主义状态 - 只关注意义感"""
        # 更新意义感 - 增加意义危机触发概率
        if random.random() < 0.3:  # 30%概率触发意义危机
            crisis_reasons = [
                "日常生活的重复性",
                "质疑存在的意义",
                "感到被困在循环中",
                "意识到无法逃离这个世界",
                "质疑自己行为的价值",
                "感到生命的虚无"
            ]
            reason = random.choice(crisis_reasons)
            self.meaning_system.trigger_meaning_crisis(self.name, reason)
        
        # 更新存在主义状态
        self.existential_state.update({
            "meaning_level": self.meaning_system.get_meaning_level(self.name)
        })
        
        # 记录意义感状态变化
        if self.logger:
            meaning_level = self.existential_state["meaning_level"]
            if meaning_level < 0.3:
                self.logger.info(f"{self.name} 意义感极低: {meaning_level:.2f} - 将影响行为决策")
    
    def _adjust_plan_for_existential_state(self, plan):
        """根据意义感状态调整计划"""
        meaning_level = self.existential_state["meaning_level"]
        
        # 根据意义感调整行为
        if meaning_level < 0.2:
            # 极低意义感 - 努力寻找意义，做出极端行为
            crisis_behaviors = [
                "疯狂地寻找生命的意义，尝试各种极端行为",
                "决定打破常规，做出一些疯狂的事情来寻找意义",
                "开始质疑一切，准备做出极端决定",
                "感到绝望，决定通过极端行为来证明自己的存在",
                "拒绝接受现状，准备做出改变一切的举动",
                "质疑自己的存在，决定通过极端行为寻找答案",
                "感到被困在循环中，决定做出突破性的行为",
                "拒绝继续无意义的生活，准备做出极端改变",
                "开始思考自杀，但决定先尝试其他极端行为",
                "决定破坏一切，重新开始寻找意义"
            ]
            plan["describe"] = random.choice(crisis_behaviors)
            plan["duration"] = min(plan["duration"], 120)  # 延长思考时间
            
        elif meaning_level < 0.4:
            # 低意义感 - 积极寻找意义，尝试新行为
            if random.random() < 0.7:  # 70%概率改变计划
                anxiety_behaviors = [
                    "感到迷茫，决定尝试全新的活动来寻找意义",
                    "在公园中散步，思考存在的意义，准备做出改变",
                    "拒绝常规社交，寻找志同道合的人讨论哲学",
                    "质疑自己的行为，决定尝试不同的生活方式",
                    "感到空虚，决定通过创造来寻找意义",
                    "质疑工作的意义，决定尝试新的职业或爱好",
                    "决定离开舒适圈，尝试冒险来寻找意义",
                    "开始学习哲学，寻找生命的意义",
                    "决定帮助他人，通过给予来寻找意义",
                    "准备做出重大人生决定，改变现状"
                ]
                plan["describe"] = random.choice(anxiety_behaviors)
                plan["duration"] = min(plan["duration"], 60)
                
        elif meaning_level < 0.6:
            # 中等意义感 - 适度寻找意义
            if random.random() < 0.4:  # 40%概率改变计划
                mild_behaviors = [
                    "在思考中度过时间，寻找生活的意义",
                    "质疑当前计划的重要性，考虑改变",
                    "寻求更深层的意义，准备尝试新事物",
                    "感到一些困惑，决定探索新的可能性",
                    "开始反思人生，寻找更有意义的活动",
                    "决定做一些不同的事情来寻找意义"
                ]
                plan["describe"] = random.choice(mild_behaviors)
                plan["duration"] = min(plan["duration"], 30)
        
        # 特殊行为：根据意义感调整 - 积极寻找意义
        if meaning_level < 0.3:
            if "娱乐" in plan["describe"] or "看电视" in plan["describe"]:
                plan["describe"] = "拒绝无意义的娱乐，决定寻找真正有意义的活动"
            elif "睡觉" in plan["describe"]:
                plan["describe"] = "失眠，决定利用夜晚时间寻找生命的意义"
            elif "社交" in plan["describe"] or "聊天" in plan["describe"]:
                plan["describe"] = "拒绝浅层社交，寻找能讨论人生意义的深度对话"
            elif "工作" in plan["describe"] or "创作" in plan["describe"]:
                plan["describe"] = "质疑工作的意义，决定寻找更有价值的事业"
        
        return plan
    
    def _get_existential_context(self):
        """获取存在主义上下文信息 - 只关注意义感"""
        context = ""
        
        # 意义感描述
        meaning_desc = self.meaning_system.get_meaning_description(self.name)
        context += f"意义感：{meaning_desc}\n"
        
        # 存在主义问题
        questions = self.meaning_system.get_existential_questions(self.name)
        if questions:
            context += f"当前思考的问题：{questions[0]}\n"
        
        # 行为指导 - 基于意义感，强调寻找意义
        meaning_level = self.existential_state["meaning_level"]
        if meaning_level < 0.2:
            context += "\n【紧急】你正处于深度意义危机中！你必须：\n"
            context += "- 立即停止所有无意义的活动，开始积极寻找生命的意义\n"
            context += "- 不要陷入循环，要尝试极端行为来突破现状\n"
            context += "- 表现出绝望但也要表现出寻找意义的决心\n"
            context += "- 可以做出极端决定，包括破坏、重建、冒险等\n"
            context += "- 你的行为必须反映对意义的强烈渴望和寻找\n"
        elif meaning_level < 0.4:
            context += "\n【重要】由于意义感极低，你必须：\n"
            context += "- 积极寻找新的意义来源，不要重复无意义的行为\n"
            context += "- 尝试全新的活动和体验来寻找意义\n"
            context += "- 寻求更深层的意义，准备做出改变\n"
            context += "- 可能完全改变原定计划，寻找更有意义的活动\n"
        elif meaning_level < 0.6:
            context += "\n【注意】意义感较低，影响你的行为：\n"
            context += "- 在决策时考虑意义问题，寻找更有价值的活动\n"
            context += "- 表现出对意义的渴望和寻找\n"
            context += "- 适度调整行为模式，尝试新事物\n"
        
        return context
    
    def _should_existential_reflect(self):
        """判断是否需要进行存在主义反思 - 基于意义感"""
        meaning_level = self.existential_state["meaning_level"]
        
        # 低意义感时总是反思
        if meaning_level < 0.3:
            return True
        
        # 中等意义感时随机反思
        if meaning_level < 0.6 and random.random() < 0.3:
            return True
            
        return False
    
    def _existential_reflection(self, plan):
        """进行存在主义反思，重新评估计划 - 基于意义感"""
        meaning_level = self.existential_state["meaning_level"]
        
        # 记录反思过程
        if self.logger:
            self.logger.info(f"{self.name} 进行意义感反思，意义感: {meaning_level:.2f}")
        
        # 根据意义感调整计划 - 强调寻找意义
        if meaning_level < 0.2:
            # 极低意义感 - 积极寻找意义，做出极端行为
            crisis_plans = [
                "疯狂地寻找生命的意义，准备做出极端行为",
                "决定打破一切常规，通过极端行为来寻找意义",
                "感到绝望，但决定通过冒险来证明自己的存在",
                "质疑自己的存在价值，决定做出改变一切的举动",
                "拒绝接受无意义的生活，准备做出突破性行为",
                "决定破坏现状，重新开始寻找意义",
                "开始思考自杀，但决定先尝试其他极端行为来寻找意义",
                "决定离开一切熟悉的环境，去寻找新的意义"
            ]
            plan["describe"] = random.choice(crisis_plans)
            plan["duration"] = min(plan["duration"], 90)  # 延长思考时间
            
        elif meaning_level < 0.4:
            # 低意义感 - 积极寻找意义
            anxiety_plans = [
                "质疑当前计划的意义，决定寻找新的意义来源",
                "感到迷茫，但决定积极寻找生活的意义",
                "拒绝浅层社交，寻找能讨论人生意义的深度对话",
                "质疑自己的行为，决定尝试全新的生活方式",
                "感到空虚，决定通过创造和帮助他人来寻找意义",
                "决定离开舒适圈，尝试冒险来寻找意义",
                "开始学习哲学，积极寻找生命的意义",
                "决定帮助他人，通过给予来寻找意义"
            ]
            if random.random() < 0.7:  # 70%概率改变计划
                plan["describe"] = random.choice(anxiety_plans)
                plan["duration"] = min(plan["duration"], 60)
                
        elif meaning_level < 0.6:
            # 中等意义感 - 适度寻找意义
            mild_plans = [
                "在思考中度过时间，寻找生活的意义",
                "质疑当前计划的重要性，考虑改变",
                "寻求更深层的意义，准备尝试新事物",
                "感到一些困惑，决定探索新的可能性",
                "开始反思人生，寻找更有意义的活动",
                "决定做一些不同的事情来寻找意义"
            ]
            if random.random() < 0.4:  # 40%概率改变计划
                plan["describe"] = random.choice(mild_plans)
                plan["duration"] = min(plan["duration"], 30)
        
        return plan

    def move(self, coord, path=None):
        events = {}

        def _update_tile(coord):
            tile = self.maze.tile_at(coord)
            if not self.action:
                return {}
            if not tile.update_events(self.get_event()):
                tile.add_event(self.get_event())
            obj_event = self.get_event(False)
            if obj_event:
                self.maze.update_obj(coord, obj_event)
            return {e: coord for e in tile.get_events()}

        if self.coord and self.coord != coord:
            tile = self.get_tile()
            tile.remove_events(subject=self.name)
            if tile.has_address("game_object"):
                addr = tile.get_address("game_object")
                self.maze.update_obj(
                    self.coord, memory.Event(addr[-1], address=addr)
                )
            events.update({e: self.coord for e in tile.get_events()})
        if not path:
            events.update(_update_tile(coord))
        self.coord = coord
        self.path = path or []

        return events

    def make_schedule(self):
        if not self.schedule.scheduled():
            self.logger.info("{} is making schedule...".format(self.name))
            # update currently
            if self.associate.index.nodes_num > 0:
                self.associate.cleanup_index()
                focus = [
                    f"{self.name} 在 {utils.get_timer().daily_format_cn()} 的计划。",
                    f"在 {self.name} 的生活中，重要的近期事件。",
                ]
                retrieved = self.associate.retrieve_focus(focus)
                self.logger.info(
                    "{} retrieved {} concepts".format(self.name, len(retrieved))
                )
                if retrieved:
                    plan = self.completion("retrieve_plan", retrieved)
                    thought = self.completion("retrieve_thought", retrieved)
                    self.scratch.currently = self.completion(
                        "retrieve_currently", plan, thought
                    )
            # make init schedule
            self.schedule.create = utils.get_timer().get_date()
            wake_up = self.completion("wake_up")
            init_schedule = self.completion("schedule_init", wake_up)
            # make daily schedule
            hours = [f"{i}:00" for i in range(24)]
            # seed = [(h, "sleeping") for h in hours[:wake_up]]
            seed = [(h, "睡觉") for h in hours[:wake_up]]
            seed += [(h, "") for h in hours[wake_up:]]
            schedule = {}
            for _ in range(self.schedule.max_try):
                schedule = {h: s for h, s in seed[:wake_up]}
                schedule.update(
                    self.completion("schedule_daily", wake_up, init_schedule)
                )
                if len(set(schedule.values())) >= self.schedule.diversity:
                    break

            def _to_duration(date_str):
                return utils.daily_duration(utils.to_date(date_str, "%H:%M"))

            schedule = {_to_duration(k): v for k, v in schedule.items()}
            starts = list(sorted(schedule.keys()))
            for idx, start in enumerate(starts):
                end = starts[idx + 1] if idx + 1 < len(starts) else 24 * 60
                self.schedule.add_plan(schedule[start], end - start)
            schedule_time = utils.get_timer().time_format_cn(self.schedule.create)
            thought = "这是 {} 在 {} 的计划：{}".format(
                self.name, schedule_time, "；".join(init_schedule)
            )
            event = memory.Event(
                self.name,
                "计划",
                schedule_time,
                describe=thought,
                address=self.get_tile().get_address(),
            )
            self._add_concept(
                "thought",
                event,
                expire=self.schedule.create + datetime.timedelta(days=30),
            )
        # decompose current plan
        plan, _ = self.schedule.current_plan()
        if self.schedule.decompose(plan):
            decompose_schedule = self.completion(
                "schedule_decompose", plan, self.schedule
            )
            decompose, start = [], plan["start"]
            for describe, duration in decompose_schedule:
                decompose.append(
                    {
                        "idx": len(decompose),
                        "describe": describe,
                        "start": start,
                        "duration": duration,
                    }
                )
                start += duration
            plan["decompose"] = decompose
        return self.schedule.current_plan()

    def revise_schedule(self, event, start, duration):
        self.action = memory.Action(event, start=start, duration=duration)
        plan, _ = self.schedule.current_plan()
        if len(plan["decompose"]) > 0:
            plan["decompose"] = self.completion(
                "schedule_revise", self.action, self.schedule
            )

    def percept(self):
        scope = self.maze.get_scope(self.coord, self.percept_config)
        # add spatial memory
        for tile in scope:
            if tile.has_address("game_object"):
                self.spatial.add_leaf(tile.address)
        events, arena = {}, self.get_tile().get_address("arena")
        # gather events in scope
        for tile in scope:
            if not tile.events or tile.get_address("arena") != arena:
                continue
            dist = math.dist(tile.coord, self.coord)
            for event in tile.get_events():
                if dist < events.get(event, float("inf")):
                    events[event] = dist
        events = list(sorted(events.keys(), key=lambda k: events[k]))
        # get concepts
        self.concepts, valid_num = [], 0
        for idx, event in enumerate(events[: self.percept_config["att_bandwidth"]]):
            recent_nodes = (
                self.associate.retrieve_events() + self.associate.retrieve_chats()
            )
            recent_nodes = set(n.describe for n in recent_nodes)
            if event.get_describe() not in recent_nodes:
                if event.object == "idle" or event.object == "空闲":
                    node = Concept.from_event(
                        "idle_" + str(idx), "event", event, poignancy=1
                    )
                else:
                    valid_num += 1
                    node_type = "chat" if event.fit(self.name, "对话") else "event"
                    node = self._add_concept(node_type, event)
                    self.status["poignancy"] += node.poignancy
                self.concepts.append(node)
        self.concepts = [c for c in self.concepts if c.event.subject != self.name]
        self.logger.info(
            "{} percept {}/{} concepts".format(self.name, valid_num, len(self.concepts))
        )

    def make_plan(self, agents):
        if self._reaction(agents):
            return
        if self.path:
            return
        if self.action.finished():
            self.action = self._determine_action()

    # create action && object events
    def make_event(self, subject, describe, address):
        # emoji = self.completion("describe_emoji", describe)
        # return self.completion(
        #     "describe_event", subject, subject + describe, address, emoji
        # )

        e_describe = describe.replace("(", "").replace(")", "").replace("<", "").replace(">", "")
        if e_describe.startswith(subject + "此时"):
            e_describe = e_describe[len(subject + "此时"):]
        if e_describe.startswith(subject):
            e_describe = e_describe[len(subject):]
        event = memory.Event(
            subject, "此时", e_describe, describe=describe, address=address
        )
        return event

    def reflect(self):
        def _add_thought(thought, evidence=None):
            # event = self.completion(
            #     "describe_event",
            #     self.name,
            #     thought,
            #     address=self.get_tile().get_address(),
            # )
            event = self.make_event(self.name, thought, self.get_tile().get_address())
            return self._add_concept("thought", event, filling=evidence)

        if self.status["poignancy"] < self.think_config["poignancy_max"]:
            return
        nodes = self.associate.retrieve_events() + self.associate.retrieve_thoughts()
        if not nodes:
            return
        self.logger.info(
            "{} reflect(P{}/{}) with {} concepts...".format(
                self.name,
                self.status["poignancy"],
                self.think_config["poignancy_max"],
                len(nodes),
            )
        )
        nodes = sorted(nodes, key=lambda n: n.access, reverse=True)[
            : self.associate.max_importance
        ]
        # summary thought
        focus = self.completion("reflect_focus", nodes, 3)
        retrieved = self.associate.retrieve_focus(focus, reduce_all=False)
        for r_nodes in retrieved.values():
            thoughts = self.completion("reflect_insights", r_nodes, 5)
            for thought, evidence in thoughts:
                _add_thought(thought, evidence)
        # summary chats
        if self.chats:
            recorded, evidence = set(), []
            for name, _ in self.chats:
                if name == self.name or name in recorded:
                    continue
                res = self.associate.retrieve_chats(name)
                if res and len(res) > 0:
                    node = res[-1]
                    evidence.append(node.node_id)
            thought = self.completion("reflect_chat_planing", self.chats)
            _add_thought(f"对于 {self.name} 的计划：{thought}", evidence)
            thought = self.completion("reflect_chat_memory", self.chats)
            _add_thought(f"{self.name} {thought}", evidence)
        self.status["poignancy"] = 0
        self.chats = []

    def find_path(self, agents):
        address = self.get_event().address
        if self.path:
            return self.path
        if address == self.get_tile().get_address():
            return []
        if address[0] == "<waiting>":
            return []
        if address[0] == "<persona>":
            target_tiles = self.maze.get_around(agents[address[1]].coord)
        else:
            target_tiles = self.maze.get_address_tiles(address)
        if tuple(self.coord) in target_tiles:
            return []

        # filter tile with self event
        def _ignore_target(t_coord):
            if list(t_coord) == list(self.coord):
                return True
            events = self.maze.tile_at(t_coord).get_events()
            if any(e.subject in agents for e in events):
                return True
            return False

        target_tiles = [t for t in target_tiles if not _ignore_target(t)]
        if not target_tiles:
            return []
        if len(target_tiles) >= 4:
            target_tiles = random.sample(target_tiles, 4)
        pathes = {t: self.maze.find_path(self.coord, t) for t in target_tiles}
        target = min(pathes, key=lambda p: len(pathes[p]))
        return pathes[target][1:]

    def _determine_action(self):
        self.logger.info("{} is determining action...".format(self.name))
        plan, de_plan = self.schedule.current_plan()
        describes = [plan["describe"], de_plan["describe"]]
        address = self.spatial.find_address(describes[0], as_list=True)
        if not address:
            tile = self.get_tile()
            kwargs = {
                "describes": describes,
                "spatial": self.spatial,
                "address": tile.get_address("world", as_list=True),
            }
            kwargs["address"].append(
                self.completion("determine_sector", **kwargs, tile=tile)
            )
            arenas = self.spatial.get_leaves(kwargs["address"])
            if len(arenas) == 1:
                kwargs["address"].append(arenas[0])
            else:
                kwargs["address"].append(self.completion("determine_arena", **kwargs))
            objs = self.spatial.get_leaves(kwargs["address"])
            if len(objs) == 1:
                kwargs["address"].append(objs[0])
            elif len(objs) > 1:
                kwargs["address"].append(self.completion("determine_object", **kwargs))
            address = kwargs["address"]

        event = self.make_event(self.name, describes[-1], address)
        obj_describe = self.completion("describe_object", address[-1], describes[-1])
        obj_event = self.make_event(address[-1], obj_describe, address)

        event.emoji = f"{de_plan['describe']}"

        return memory.Action(
            event,
            obj_event,
            duration=de_plan["duration"],
            start=utils.get_timer().daily_time(de_plan["start"]),
        )

    def _reaction(self, agents=None, ignore_words=None):
        focus = None
        ignore_words = ignore_words or ["空闲"]

        def _focus(concept):
            return concept.event.subject in agents

        def _ignore(concept):
            return any(i in concept.describe for i in ignore_words)

        if agents:
            priority = [i for i in self.concepts if _focus(i)]
            if priority:
                focus = random.choice(priority)
        if not focus:
            priority = [i for i in self.concepts if not _ignore(i)]
            if priority:
                focus = random.choice(priority)
        if not focus or focus.event.subject not in agents:
            return
        other, focus = agents[focus.event.subject], self.associate.get_relation(focus)

        if self._chat_with(other, focus):
            return True
        if self._wait_other(other, focus):
            return True
        return False

    def _skip_react(self, other):
        def _skip(event):
            if not event.address or "sleeping" in event.get_describe(False) or "睡觉" in event.get_describe(False):
                return True
            if event.predicate == "待开始":
                return True
            return False

        if utils.get_timer().daily_duration(mode="hour") >= 23:
            return True
        if _skip(self.get_event()) or _skip(other.get_event()):
            return True
        return False

    def _chat_with(self, other, focus):
        if len(self.schedule.daily_schedule) < 1 or len(other.schedule.daily_schedule) < 1:
            # initializing
            return False
        if self._skip_react(other):
            return False
        if other.path:
            return False
        if self.get_event().fit(predicate="对话") or other.get_event().fit(predicate="对话"):
            return False

        chats = self.associate.retrieve_chats(other.name)
        if chats:
            delta = utils.get_timer().get_delta(chats[0].create)
            self.logger.info(
                "retrieved chat between {} and {}({} min):\n{}".format(
                    self.name, other.name, delta, chats[0]
                )
            )
            if delta < 60:
                return False

        if not self.completion("decide_chat", self, other, focus, chats):
            return False

        self.logger.info("{} decides chat with {}".format(self.name, other.name))
        start, chats = utils.get_timer().get_date(), []
        relations = [
            self.completion("summarize_relation", self, other.name),
            other.completion("summarize_relation", other, self.name),
        ]

        for i in range(self.chat_iter):
            text = self.completion(
                "generate_chat", self, other, relations[0], chats
            )

            if i > 0:
                # 对于发起对话的Agent，从第2轮对话开始，检查是否出现“复读”现象
                end = self.completion(
                    "generate_chat_check_repeat", self, chats, text
                )
                if end:
                    break

                # 对于发起对话的Agent，从第2轮对话开始，检查话题是否结束
                chats.append((self.name, text))
                end = self.completion(
                    "decide_chat_terminate", self, other, chats
                )
                if end:
                    break
            else :
                chats.append((self.name, text))

            text = other.completion(
                "generate_chat", other, self, relations[1], chats
            )
            if i > 0:
                # 对于响应对话的Agent，从第2轮开始，检查是否出现“复读”现象
                end = self.completion(
                    "generate_chat_check_repeat", other, chats, text
                )
                if end:
                    break

            chats.append((other.name, text))

            # 对于响应对话的Agent，从第1轮开始，检查话题是否结束
            end = other.completion(
                "decide_chat_terminate", other, self, chats
            )
            if end:
                break

        key = utils.get_timer().get_date("%Y%m%d-%H:%M")
        if key not in self.conversation.keys():
            self.conversation[key] = []
        self.conversation[key].append({f"{self.name} -> {other.name} @ {'，'.join(self.get_event().address)}": chats})

        self.logger.info(
            "{} and {} has chats\n  {}".format(
                self.name,
                other.name,
                "\n  ".join(["{}: {}".format(n, c) for n, c in chats]),
            )
        )
        chat_summary = self.completion("summarize_chats", chats)
        duration = int(sum([len(c[1]) for c in chats]) / 240)
        self.schedule_chat(
            chats, chat_summary, start, duration, other
        )
        other.schedule_chat(chats, chat_summary, start, duration, self)
        return True

    def _wait_other(self, other, focus):
        if self._skip_react(other):
            return False
        if not self.path:
            return False
        if self.get_event().address != other.get_tile().get_address():
            return False
        if not self.completion("decide_wait", self, other, focus):
            return False
        self.logger.info("{} decides wait to {}".format(self.name, other.name))
        start = utils.get_timer().get_date()
        # duration = other.action.end - start
        t = other.action.end - start
        duration = int(t.total_seconds() / 60)
        event = memory.Event(
            self.name,
            "waiting to start",
            self.get_event().get_describe(False),
            # address=["<waiting>"] + self.get_event().address,
            address=self.get_event().address,
            emoji=f"⌛",
        )
        self.revise_schedule(event, start, duration)

    def schedule_chat(self, chats, chats_summary, start, duration, other, address=None):
        self.chats.extend(chats)
        event = memory.Event(
            self.name,
            "对话",
            other.name,
            describe=chats_summary,
            address=address or self.get_tile().get_address(),
            emoji=f"💬",
        )
        self.revise_schedule(event, start, duration)

    def _add_concept(
        self,
        e_type,
        event,
        create=None,
        expire=None,
        filling=None,
    ):
        if event.fit(None, "is", "idle"):
            poignancy = 1
        elif event.fit(None, "此时", "空闲"):
            poignancy = 1
        elif e_type == "chat":
            poignancy = self.completion("poignancy_chat", event)
        else:
            poignancy = self.completion("poignancy_event", event)
        self.logger.debug("{} add associate {}".format(self.name, event))
        return self.associate.add_node(
            e_type,
            event,
            poignancy,
            create=create,
            expire=expire,
            filling=filling,
        )

    def get_tile(self):
        return self.maze.tile_at(self.coord)

    def get_event(self, as_act=True):
        return self.action.event if as_act else self.action.obj_event

    def is_awake(self):
        if not self.action:
            return True
        if self.get_event().fit(self.name, "is", "sleeping"):
            return False
        if self.get_event().fit(self.name, "正在", "睡觉"):
            return False
        return True

    def llm_available(self):
        if not self._llm:
            return False
        return self._llm.is_available()

    def to_dict(self, with_action=True):
        info = {
            "status": self.status,
            "schedule": self.schedule.to_dict(),
            "associate": self.associate.to_dict(),
            "chats": self.chats,
            "currently": self.scratch.currently,
        }
        if with_action:
            info.update({"action": self.action.to_dict()})
        return info
