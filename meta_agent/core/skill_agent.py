"""
SkillAgent - 技能驱动的Agent核心
通过动态加载Claude Skills来扩展能力
"""
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import uuid

from ..state.agent_state import AgentState, create_initial_state
from ..state.state_manager import StateManager
from ..core.context_manager import ContextManager
from ..core.memory import Memory
from ..skills.skill_registry import SkillRegistry
from ..skills.skill_loader import SkillLoader
from ..tools.tool_manager import ToolManager, get_tool_manager
from ..utils.prompt_parser import PromptParser, ParsedPrompt
from .skill_executor import SkillExecutor, SkillExecutionResult


logger = logging.getLogger(__name__)


class SkillAgent:
    """
    技能驱动的Agent
    根据用户prompt动态加载和执行技能
    """
    
    def __init__(
        self,
        skill_registry: Optional[SkillRegistry] = None,
        tool_manager: Optional[ToolManager] = None,
        session_id: Optional[str] = None
    ):
        """
        初始化SkillAgent
        
        Args:
            skill_registry: 技能注册表（可选，默认创建新实例）
            tool_manager: 工具管理器（可选，默认使用全局实例）
            session_id: 会话ID（可选，默认生成新ID）
        """
        # 初始化组件
        self.skill_registry = skill_registry or SkillRegistry()
        self.skill_loader = SkillLoader(self.skill_registry.skills_dir)
        self.tool_manager = tool_manager or get_tool_manager()
        
        # 初始化执行器
        self.skill_executor = SkillExecutor(
            self.skill_registry,
            self.skill_loader,
            self.tool_manager
        )
        
        # 初始化解析器
        self.prompt_parser = PromptParser()
        
        # 会话管理
        self.session_id = session_id or str(uuid.uuid4())
        self.state: Optional[AgentState] = None
        self.state_manager: Optional[StateManager] = None
        self.context_manager: Optional[ContextManager] = None
        
        # 执行状态
        self.is_running = False
        self.current_task: Optional[str] = None
        
        logger.debug(f"SkillAgent initialized with session_id: {self.session_id}")
    
    def initialize_session(self, user_prompt: str) -> AgentState:
        """
        初始化会话
        
        Args:
            user_prompt: 用户输入的prompt
            
        Returns:
            初始化的AgentState
        """
        # 创建初始状态
        self.state = create_initial_state(user_prompt, self.session_id)
        
        # 初始化状态管理器
        self.state_manager = StateManager()
        
        # 初始化上下文管理器
        self.context_manager = ContextManager(self.state)
        
        # 初始化Memory模块
        self.memory = Memory(session_id=self.session_id)

        # 解析prompt
        parsed_prompt = self.prompt_parser.parse(user_prompt)
        
        # 更新状态
        self.state["task_goal"] = parsed_prompt.task_goal
        
        # 加载技能元数据到共享知识库
        all_skills = self.skill_registry.get_all_skills()
        skill_metadata_cache = {
            skill.name: f"{skill.name}: {skill.description}"
            for skill in all_skills
        }
        self.state["shared_knowledge"]["skill_metadata"] = skill_metadata_cache
        
        # 加载工具列表
        self.state["available_tools"] = self.tool_manager.get_all_tool_names()
        
        logger.debug(f"Session initialized: {self.session_id}")
        return self.state
    
    async def process_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """
        处理用户prompt（使用ReAct模式）

        Args:
            user_prompt: 用户输入

        Returns:
            处理结果
        """
        try:
            self.is_running = True

            # 初始化会话
            if not self.state:
                self.initialize_session(user_prompt)

            # 使用ReAct模式处理
            result = await self._react_loop(user_prompt)

            self.is_running = False
            return result

        except Exception as e:
            logger.error(f"Failed to process prompt: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False
            return {
                "success": False,
                "error": str(e)
            }

    async def process_with_plan_execute(
        self,
        user_prompt: str,
        save_to_local: bool = False
    ) -> Dict[str, Any]:
        """
        使用 Plan-Execute 模式处理用户prompt

        Args:
            user_prompt: 用户输入
            save_to_local: 是否保存到本地

        Returns:
            处理结果
        """
        try:
            self.is_running = True
            logger.debug(f"Processing with Plan-Execute mode: {user_prompt}")

            # 创建 Plan-Execute 状态
            from meta_agent.state.agent_state import create_plan_execute_state
            initial_state = create_plan_execute_state(user_prompt, self.session_id)
            initial_state["save_to_local"] = save_to_local

            # 构建 Plan-Execute 图
            from meta_agent.core.graph_builder import GraphBuilder
            graph_builder = GraphBuilder(self)
            graph = graph_builder.build_graph(mode="plan_execute")

            # 执行图
            logger.debug("Executing Plan-Execute graph")
            final_state = await graph.ainvoke(initial_state)

            # 提取结果
            result = {
                "success": len(final_state.get("errors", [])) == 0,
                "final_output": final_state.get("final_output", ""),
                "plan": final_state.get("plan", []),
                "step_results": final_state.get("step_results", {}),
                "qa_report": final_state.get("qa_report", ""),  # 添加 QA 报告
                "output_directory": final_state.get("output_directory", ""),
                "errors": final_state.get("errors", []),
                "session_id": self.session_id,
                "mode": final_state.get("mode", "plan_execute")  # 从 final_state 获取实际模式
            }

            self.is_running = False
            logger.debug("Plan-Execute processing completed")
            return result

        except Exception as e:
            logger.error(f"Failed to process with Plan-Execute: {e}", exc_info=True)
            self.is_running = False
            return {
                "success": False,
                "error": str(e),
                "mode": "plan_execute"
            }

    async def _react_loop(self, user_prompt: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        ReAct循环：Reasoning + Acting

        Args:
            user_prompt: 用户输入
            max_iterations: 最大迭代次数

        Returns:
            执行结果
        """
        from meta_agent.core.llm_manager import LLMManager
        from meta_agent.config.agent_prompts import get_agent_prompt

        llm = LLMManager()
        agent_prompt = get_agent_prompt("social_media")

        # 添加用户消息到Memory
        self.memory.add_message("user", user_prompt)

        # Layer 1: 只加载技能元数据（轻量级）
        skill_name = "content-generate"
        metadata = self.skill_loader.load_skill_metadata(skill_name)

        logger.debug(f"Loading skill metadata (Layer 1) for: {skill_name}")

        skill_info_section = ""
        if metadata:
            # 只展示基础元数据，不加载完整的 Summary
            metadata_lines = []
            if metadata.name:
                metadata_lines.append(f"**技能名称**: {metadata.name}")
            if metadata.description:
                metadata_lines.append(f"**描述**: {metadata.description}")
            if metadata.category:
                metadata_lines.append(f"**类别**: {metadata.category}")
            if metadata.version:
                metadata_lines.append(f"**版本**: {metadata.version}")
            if metadata.author:
                metadata_lines.append(f"**作者**: {metadata.author}")
            if metadata.tags:
                tags_str = ", ".join(metadata.tags)
                metadata_lines.append(f"**标签**: {tags_str}")

            metadata_text = "\n".join(metadata_lines) if metadata_lines else ""

            skill_info_section = f"""

---

## 当前可用技能

{metadata_text}

**提示**：如果用户需要生成社交媒体内容，我会自动加载详细的技能指南。

---
"""
            logger.debug(f"Skill metadata loaded (Layer 1), length: {len(skill_info_section)}")
        else:
            logger.warning("Failed to load skill metadata")

        # 构建ReAct系统提示词（包含历史上下文）
        recent_context = self.memory.get_recent_context(max_messages=6)
        context_section = ""
        if recent_context:
            context_section = f"\n\n## 最近对话上下文\n\n{recent_context}\n\n---\n"

        react_prompt = f"""{agent_prompt}{skill_info_section}{context_section}

---

## 执行指令

请严格按照上述技能详情中的 SOP 执行。

请开始执行。"""

        # 从Memory获取对话历史
        conversation_history = self.memory.get_conversation_history(
            include_system=False,
            last_n=10
        )

        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.debug(f"ReAct iteration {iteration}/{max_iterations}")

            # 构建消息
            if iteration == 1:
                current_message = user_prompt
            else:
                current_message = "请继续"

            # 调用LLM进行推理
            response = await llm.generate_with_history(
                system_prompt=react_prompt,
                messages=conversation_history + [{"role": "user", "content": current_message}]
            )

            # 解析响应
            action_result = self._parse_react_response(response)

            # 记录到Memory
            if iteration > 1:
                self.memory.add_message("user", current_message)
            self.memory.add_message("assistant", response)

            # 更新对话历史
            conversation_history = self.memory.get_conversation_history(
                include_system=False,
                last_n=10
            )

            # 根据Action执行
            if action_result["action"] == "CHAT":
                return {
                    "success": True,
                    "orchestrator_response": action_result.get("response", response),
                    "mode": "chat",
                    "session_id": self.session_id
                }

            elif action_result["action"] == "GENERATE_CONTENT":
                # 执行内容生成
                return await self._execute_content_generation(
                    action_result,
                    response,
                    user_prompt  # 传递用户原始输入
                )

            elif action_result["action"] == "FINISH":
                return {
                    "success": True,
                    "orchestrator_response": response,
                    "mode": "finish",
                    "session_id": self.session_id
                }

            # 如果没有明确的Action，继续循环
            logger.warning(f"No clear action in iteration {iteration}, continuing...")

        # 达到最大迭代次数
        return {
            "success": False,
            "error": "达到最大迭代次数",
            "orchestrator_response": conversation_history[-1]["content"] if conversation_history else ""
        }

    def _parse_react_response(self, response: str) -> Dict[str, Any]:
        """
        解析ReAct响应

        Args:
            response: LLM的响应

        Returns:
            解析结果
        """
        import re

        result = {
            "action": "UNKNOWN",
            "thought": "",
            "response": response
        }

        # 提取Thought
        thought_match = re.search(r'Thought[：:]\s*(.+?)(?=Action|$)', response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 提取Action
        action_match = re.search(r'Action[：:]\s*(\w+)', response, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).upper()

        # 如果包含Worker指令，判断为GENERATE_CONTENT
        if "[[EXECUTE_WORKER:" in response or "@" in response and "Expert" in response:
            result["action"] = "GENERATE_CONTENT"

        # 如果响应很短且没有特殊标记，判断为CHAT
        if result["action"] == "UNKNOWN" and len(response) < 200 and "[[" not in response:
            result["action"] = "CHAT"

        # 如果是CHAT action，提取Response内容（如果有的话）
        if result["action"] == "CHAT":
            response_match = re.search(r'Response[：:]\s*(.+)', response, re.DOTALL | re.IGNORECASE)
            if response_match:
                result["response"] = response_match.group(1).strip()
            else:
                # 如果没有Response标记，移除Thought和Action标记后的内容作为response
                cleaned = re.sub(r'```[\s\S]*?```', '', response)
                cleaned = re.sub(r'(Thought|Action|Workers|Input|Response)[：:].*?\n', '', cleaned)
                cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
                if cleaned:
                    result["response"] = cleaned
                else:
                    result["response"] = response

        return result

    async def _execute_content_generation(
        self,
        action_result: Dict[str, Any],
        orchestrator_response: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        执行内容生成

        Args:
            action_result: Action解析结果
            orchestrator_response: Orchestrator的完整响应
            user_prompt: 用户的原始输入

        Returns:
            生成结果
        """
        from meta_agent.core.graph_builder import GraphBuilder

        # Layer 2: 当确认需要生成内容时，加载完整的 Skill Summary
        skill_name = "content-generate"
        skill_data = self.skill_loader.load_skill_with_metadata(skill_name)

        logger.debug(f"Loading skill details (Layer 2) for: {skill_name}")

        if skill_data:
            skill_summary = skill_data.get("summary", "")
            logger.debug(f"Skill summary loaded (Layer 2), length: {len(skill_summary)}")

            # 将完整的 Summary 添加到 shared_knowledge 中供 Worker 使用
            if not hasattr(self, 'state') or self.state is None:
                self.state = {}
            if 'shared_knowledge' not in self.state:
                self.state['shared_knowledge'] = {}
            self.state['shared_knowledge']['skill_summary'] = skill_summary
        else:
            logger.warning("Failed to load skill summary (Layer 2)")

        # 构建content-generate图
        graph_builder = GraphBuilder(self)
        graph = graph_builder.build_graph(mode="content_generate")

        # 准备初始状态
        initial_state = {
            "user_prompt": user_prompt,  # 使用用户的原始输入
            "task_goal": "",
            "execution_history": [],
            "shared_knowledge": {
                "orchestrator_response": orchestrator_response
            },
            "active_agents": {},
            "agent_results": {},
            "current_agent_id": None,
            "current_task": None,
            "loaded_skills": [],
            "available_tools": [],
            "session_id": self.session_id,
            "iteration_count": 0,
            "errors": [],
            "worker_instructions": [],
            "worker_results": {},
            "save_to_local": False
        }

        # 执行图（跳过orchestrator节点，直接从intercept_instructions开始）
        logger.debug("Executing content generation...")
        final_state = await graph.ainvoke(initial_state)

        # 提取结果
        return {
            "success": len(final_state.get("errors", [])) == 0,
            "orchestrator_response": orchestrator_response,
            "worker_results": final_state.get("worker_results", {}),
            "output_directory": final_state.get("shared_knowledge", {}).get("output_directory", ""),
            "errors": final_state.get("errors", []),
            "session_id": self.session_id,
            "mode": "content_generate"
        }
    
    async def _select_skills(self, parsed_prompt: ParsedPrompt) -> List[str]:
        """
        根据解析的prompt选择技能
        
        Args:
            parsed_prompt: 解析后的prompt
            
        Returns:
            选中的技能名称列表
        """
        selected_skills = []
        
        # 方法1：使用prompt解析器推荐的技能
        suggested_skills = self.prompt_parser.suggest_skills(parsed_prompt)
        
        # 方法2：从技能注册表搜索
        for skill_name in suggested_skills:
            if self.skill_registry.has_skill(skill_name):
                selected_skills.append(skill_name)
        
        # 方法3：如果没有找到，使用搜索
        if not selected_skills:
            search_results = self.skill_registry.search_skills(
                parsed_prompt.task_goal
            )
            if search_results:
                # 选择最相关的技能
                selected_skills = [search_results[0].name]
        
        # 方法4：根据任务类型选择默认技能
        if not selected_skills:
            default_skill = self._get_default_skill_for_task_type(
                parsed_prompt.task_type
            )
            if default_skill:
                selected_skills = [default_skill]
        
        logger.debug(f"Selected skills: {selected_skills}")
        return selected_skills
    
    def _get_default_skill_for_task_type(self, task_type: str) -> Optional[str]:
        """
        根据任务类型获取默认技能
        
        Args:
            task_type: 任务类型
            
        Returns:
            默认技能名称
        """
        default_mapping = {
            "analysis": "code_analysis",
            "generation": "document_generation",
            "search": "web_search",
            "processing": "text_processing"
        }
        return default_mapping.get(task_type)
    
    async def _execute_skills(
        self,
        skill_names: List[str],
        parsed_prompt: ParsedPrompt
    ) -> List[SkillExecutionResult]:
        """
        执行选中的技能
        
        Args:
            skill_names: 技能名称列表
            parsed_prompt: 解析后的prompt
            
        Returns:
            执行结果列表
        """
        results = []
        
        # 准备执行上下文
        context = {
            "task_goal": parsed_prompt.task_goal,
            "task_type": parsed_prompt.task_type,
            "entities": parsed_prompt.entities,
            "constraints": parsed_prompt.constraints,
            "original_prompt": parsed_prompt.original_prompt,
            "parsed_prompt_dict": parsed_prompt.to_dict()
        }
        
        # 根据复杂度决定执行策略
        if parsed_prompt.complexity == "simple" and len(skill_names) == 1:
            # 简单任务：直接执行
            result = await self.skill_executor.execute_skill(
                skill_names[0],
                context,
                self.state
            )
            results.append(result)
        else:
            # 复杂任务：按链执行
            results = await self.skill_executor.execute_skill_chain(
                skill_names,
                context,
                self.state
            )
        
        # 更新状态
        if self.state:
            self.state["execution_history"].append({
                "timestamp": datetime.now().isoformat(),
                "skills": skill_names,
                "success": all(r.success for r in results)
            })
        
        return results
    
    def _aggregate_results(
        self,
        results: List[SkillExecutionResult],
        parsed_prompt: ParsedPrompt
    ) -> Dict[str, Any]:
        """
        整合执行结果
        
        Args:
            results: 执行结果列表
            parsed_prompt: 解析后的prompt
            
        Returns:
            整合后的结果
        """
        # 检查是否全部成功
        all_success = all(r.success for r in results)
        
        # 收集所有结果
        skill_results = {}
        errors = []
        
        for result in results:
            skill_results[result.skill_name] = result.result
            if not result.success and result.error:
                errors.append(f"{result.skill_name}: {result.error}")
        
        # 构建最终结果
        final_result = {
            "success": all_success,
            "session_id": self.session_id,
            "task_goal": parsed_prompt.task_goal,
            "task_type": parsed_prompt.task_type,
            "complexity": parsed_prompt.complexity,
            "executed_skills": [r.skill_name for r in results],
            "skill_results": skill_results,
            "timestamp": datetime.now().isoformat()
        }
        
        if errors:
            final_result["errors"] = errors
        
        # 生成摘要
        if all_success:
            final_result["summary"] = self._generate_success_summary(results, parsed_prompt)
        else:
            final_result["summary"] = f"Task partially completed. {len(errors)} error(s) occurred."
        
        return final_result
    
    def _generate_success_summary(
        self,
        results: List[SkillExecutionResult],
        parsed_prompt: ParsedPrompt
    ) -> str:
        """
        生成成功摘要
        
        Args:
            results: 执行结果列表
            parsed_prompt: 解析后的prompt
            
        Returns:
            摘要文本
        """
        skill_names = [r.skill_name for r in results]
        summary_parts = [
            f"Successfully completed task: {parsed_prompt.task_goal}",
            f"Used skills: {', '.join(skill_names)}",
            f"Complexity: {parsed_prompt.complexity}"
        ]
        
        return " | ".join(summary_parts)
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        获取会话信息
        
        Returns:
            会话信息字典
        """
        info = {
            "session_id": self.session_id,
            "is_running": self.is_running,
            "current_task": self.current_task
        }
        
        if self.state:
            info.update({
                "task_goal": self.state.get("task_goal"),
                "iteration_count": self.state.get("iteration_count", 0),
                "loaded_skills": self.state.get("loaded_skills", []),
                "available_tools_count": len(self.state.get("available_tools", []))
            })
        
        # 添加执行统计
        info["execution_stats"] = self.skill_executor.get_statistics()
        
        # 添加Memory统计
        if self.memory:
            info["memory_stats"] = self.memory.get_memory_stats()

        return info
    
    def get_available_skills(self) -> List[Dict[str, str]]:
        """
        获取可用技能列表
        
        Returns:
            技能信息列表
        """
        skills = self.skill_registry.get_all_skills()
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "category": skill.category
            }
            for skill in skills
        ]
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """
        获取可用工具列表
        
        Returns:
            工具信息列表
        """
        tool_names = self.tool_manager.get_all_tool_names()
        tools = []
        
        for tool_name in tool_names:
            schema = self.tool_manager.get_tool_schema(tool_name)
            if schema:
                # category已经是字符串（因为ToolSchema的Config设置了use_enum_values=True）
                category = schema.category if isinstance(schema.category, str) else schema.category.value
                tools.append({
                    "name": schema.name,
                    "display_name": schema.display_name,
                    "description": schema.description,
                    "category": category
                })
        
        return tools
    
    async def load_skill(self, skill_name: str) -> bool:
        """
        手动加载技能
        
        Args:
            skill_name: 技能名称
            
        Returns:
            是否加载成功
        """
        success = await self.skill_executor.load_skill(skill_name)
        
        if success and self.state:
            if skill_name not in self.state["loaded_skills"]:
                self.state["loaded_skills"].append(skill_name)
        
        return success
    
    def unload_skill(self) -> bool:
        """
        卸载当前技能
        
        Returns:
            是否卸载成功
        """
        return self.skill_executor.unload_skill()
    
    def reset_session(self):
        """重置会话"""
        self.state = None
        self.state_manager = None
        self.context_manager = None
        self.skill_executor.clear_history()

        # 重置Memory
        if self.memory:
            self.memory.clear_all()

        self.is_running = False
        self.current_task = None
        logger.debug(f"Session reset: {self.session_id}")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        获取执行历史
        
        Returns:
            执行历史列表
        """
        history = self.skill_executor.get_execution_history()
        return [result.to_dict() for result in history]
    
    async def explain_capabilities(self) -> Dict[str, Any]:
        """
        解释Agent的能力
        
        Returns:
            能力说明
        """
        return {
            "description": "SkillAgent - 技能驱动的智能Agent",
            "features": [
                "动态加载Claude Skills来扩展能力",
                "三层渐进式披露机制优化上下文使用",
                "智能prompt解析和技能匹配",
                "支持技能链式执行",
                "完整的工具管理系统",
                "Memory模块：对话历史和上下文管理"
            ],
            "available_skills": len(self.skill_registry.get_all_skills()),
            "available_tools": len(self.tool_manager.get_all_tool_names()),
            "skill_categories": list(set(
                skill.category for skill in self.skill_registry.get_all_skills()
            )),
            "tool_categories": list(set(
                schema.category.value
                for schema in [
                    self.tool_manager.get_tool_schema(name)
                    for name in self.tool_manager.get_all_tool_names()
                ]
                if schema
            ))
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取Memory统计信息

        Returns:
            Memory统计信息
        """
        if not self.memory:
            return {"error": "Memory not initialized"}

        return self.memory.get_memory_stats()

    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        获取对话历史

        Args:
            last_n: 只返回最后n条消息

        Returns:
            对话历史列表
        """
        if not self.memory:
            return []

        return self.memory.get_conversation_history(
            include_system=False,
            last_n=last_n
        )

    def clear_memory(self, memory_type: str = "all"):
        """
        清空Memory

        Args:
            memory_type: 要清空的记忆类型（short_term/long_term/all）
        """
        if not self.memory:
            logger.warning("Memory not initialized")
            return

        if memory_type == "short_term":
            self.memory.clear_short_term()
            logger.debug("Short-term memory cleared")
        elif memory_type == "long_term":
            self.memory.clear_long_term()
            logger.debug("Long-term memory cleared")
        elif memory_type == "all":
            self.memory.clear_all()
            logger.debug("All memory cleared")
        else:
            logger.warning(f"Unknown memory type: {memory_type}")

    def save_memory(self, filepath: str):
        """
        保存Memory到文件

        Args:
            filepath: 文件路径
        """
        if not self.memory:
            logger.warning("Memory not initialized")
            return

        self.memory.save_to_file(filepath)
        logger.debug(f"Memory saved to {filepath}")

    def load_memory(self, filepath: str) -> bool:
        """
        从文件加载Memory

        Args:
            filepath: 文件路径

        Returns:
            是否加载成功
        """
        try:
            loaded_memory = Memory.load_from_file(filepath)
            if loaded_memory:
                self.memory = loaded_memory
                logger.debug(f"Memory loaded from {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return False
