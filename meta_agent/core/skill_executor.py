"""
技能执行器 - 执行已加载的技能
"""
from typing import Dict, List, Optional, Any
import logging
import re
import os
from ..skills.skill_registry import SkillRegistry
from ..skills.skill_loader import SkillLoader
from ..tools.tool_manager import ToolManager
from ..state.agent_state import AgentState


logger = logging.getLogger(__name__)


class SkillExecutionResult:
    """技能执行结果"""
    
    def __init__(
        self,
        skill_name: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.skill_name = skill_name
        self.success = success
        self.result = result
        self.error = error
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "skill_name": self.skill_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


class SkillExecutor:
    """
    技能执行器
    负责执行已加载的技能，管理技能的生命周期
    """
    
    def __init__(
        self,
        skill_registry: SkillRegistry,
        skill_loader: SkillLoader,
        tool_manager: ToolManager
    ):
        """
        初始化技能执行器
        
        Args:
            skill_registry: 技能注册表
            skill_loader: 技能加载器
            tool_manager: 工具管理器
        """
        self.skill_registry = skill_registry
        self.skill_loader = skill_loader
        self.tool_manager = tool_manager
        
        # 当前激活的技能
        self.active_skill: Optional[str] = None
        
        # 技能执行历史
        self.execution_history: List[SkillExecutionResult] = []
        
        # 初始化拦截器
        from pathlib import Path
        from meta_agent.core.skill_interceptor import SkillInterceptor
        skills_dir = Path("meta_agent/skills")
        self.interceptor = SkillInterceptor(skills_dir)

        # Worker 指令缓存（LRU缓存，最多10个）
        self._worker_detail_cache: Dict[str, str] = {}
        self._cache_order: List[str] = []  # 用于LRU
        self._max_cache_size = 10

        logger.debug("SkillExecutor initialized with interceptor")
    
    async def load_skill(self, skill_name: str) -> bool:
        """
        加载技能（渐进式披露的第二层）
        
        Args:
            skill_name: 技能名称
            
        Returns:
            是否加载成功
        """
        try:
            # 检查技能是否存在
            if not self.skill_registry.has_skill(skill_name):
                logger.error(f"Skill not found: {skill_name}")
                return False
            
            # 加载技能详情
            skill_details = self.skill_loader.load_skill_details(skill_name)
            if not skill_details:
                logger.error(f"Failed to load skill details: {skill_name}")
                return False
            
            # 设置为激活技能
            self.active_skill = skill_name
            
            logger.debug(f"Skill loaded successfully: {skill_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load skill {skill_name}: {e}")
            return False
    
    def unload_skill(self) -> bool:
        """
        卸载当前技能（释放上下文）
        
        Returns:
            是否卸载成功
        """
        try:
            if self.active_skill:
                logger.debug(f"Unloading skill: {self.active_skill}")
                self.active_skill = None
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to unload skill: {e}")
            return False
    
    async def execute_skill(
        self,
        skill_name: str,
        context: Dict[str, Any],
        state: Optional[AgentState] = None
    ) -> SkillExecutionResult:
        """
        执行技能
        
        Args:
            skill_name: 技能名称
            context: 执行上下文
            state: Agent状态（可选）
            
        Returns:
            执行结果
        """
        try:
            # 加载技能（如果未加载）
            if self.active_skill != skill_name:
                success = await self.load_skill(skill_name)
                if not success:
                    return SkillExecutionResult(
                        skill_name=skill_name,
                        success=False,
                        error=f"Failed to load skill: {skill_name}"
                    )
            
            # 获取技能元数据
            skill_metadata = self.skill_registry.get_skill_metadata(skill_name)
            if not skill_metadata:
                return SkillExecutionResult(
                    skill_name=skill_name,
                    success=False,
                    error=f"Skill metadata not found: {skill_name}"
                )
            
            # 准备执行环境
            execution_context = self._prepare_execution_context(
                skill_name,
                context,
                state
            )
            
            # 执行技能逻辑
            result = await self._execute_skill_logic(
                skill_name,
                execution_context
            )
            
            # 记录执行历史
            execution_result = SkillExecutionResult(
                skill_name=skill_name,
                success=True,
                result=result,
                metadata={
                    "category": skill_metadata.category,
                    "version": skill_metadata.version
                }
            )
            self.execution_history.append(execution_result)
            
            logger.debug(f"Skill executed successfully: {skill_name}")
            return execution_result
            
        except Exception as e:
            logger.error(f"Failed to execute skill {skill_name}: {e}")
            error_result = SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                error=str(e)
            )
            self.execution_history.append(error_result)
            return error_result
    
    def _prepare_execution_context(
        self,
        skill_name: str,
        context: Dict[str, Any],
        state: Optional[AgentState]
    ) -> Dict[str, Any]:
        """
        准备执行上下文
        
        Args:
            skill_name: 技能名称
            context: 原始上下文
            state: Agent状态
            
        Returns:
            准备好的执行上下文
        """
        execution_context = {
            "skill_name": skill_name,
            "input": context,
            "tools": {},
            "state": state
        }
        
        # 获取技能所需的工具
        skill_metadata = self.skill_registry.get_skill_metadata(skill_name)
        if skill_metadata and skill_metadata.required_tools:
            for tool_name in skill_metadata.required_tools:
                tool_schema = self.tool_manager.get_tool_schema(tool_name)
                if tool_schema:
                    execution_context["tools"][tool_name] = tool_schema.to_llm_format()
        
        return execution_context
    
    async def _execute_skill_logic(
        self,
        skill_name: str,
        context: Dict[str, Any]
    ) -> Any:
        """
        执行技能的具体逻辑
        
        Args:
            skill_name: 技能名称
            context: 执行上下文
            
        Returns:
            执行结果
        """
        # 这里是技能执行的核心逻辑
        # 实际应用中，这里会调用LLM或执行具体的技能代码
        
        # 获取技能详情
        skill_details = self.skill_loader.load_skill_details(skill_name)
        if not skill_details:
            raise ValueError(f"Skill details not found: {skill_name}")
        
        # 构建执行计划
        execution_plan = {
            "skill": skill_name,
            "description": skill_details.metadata.description,
            "available_tools": list(context.get("tools", {}).keys()),
            "input_data": context.get("input", {})
        }
        
        # 模拟执行结果
        # 实际应用中，这里会：
        # 1. 调用LLM生成执行计划
        # 2. 按计划调用工具
        # 3. 收集和整合结果
        result = {
            "status": "completed",
            "execution_plan": execution_plan,
            "output": f"Skill {skill_name} executed with context: {context.get('input', {})}"
        }
        
        return result
    
    def get_execution_history(
        self,
        skill_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[SkillExecutionResult]:
        """
        获取执行历史
        
        Args:
            skill_name: 可选的技能名称过滤
            limit: 返回数量限制
            
        Returns:
            执行历史列表
        """
        history = self.execution_history
        
        # 按技能名称过滤
        if skill_name:
            history = [h for h in history if h.skill_name == skill_name]
        
        # 限制数量
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_active_skill(self) -> Optional[str]:
        """
        获取当前激活的技能
        
        Returns:
            技能名称，如果没有则返回None
        """
        return self.active_skill
    
    def clear_history(self):
        """清空执行历史"""
        self.execution_history.clear()
        logger.debug("Execution history cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        Returns:
            统计信息字典
        """
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for h in self.execution_history if h.success)
        failed_executions = total_executions - successful_executions
        
        # 按技能统计
        skill_stats = {}
        for result in self.execution_history:
            if result.skill_name not in skill_stats:
                skill_stats[result.skill_name] = {
                    "total": 0,
                    "success": 0,
                    "failed": 0
                }
            skill_stats[result.skill_name]["total"] += 1
            if result.success:
                skill_stats[result.skill_name]["success"] += 1
            else:
                skill_stats[result.skill_name]["failed"] += 1
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "active_skill": self.active_skill,
            "skill_statistics": skill_stats
        }
    
    async def execute_skill_chain(
        self,
        skill_names: List[str],
        initial_context: Dict[str, Any],
        state: Optional[AgentState] = None
    ) -> List[SkillExecutionResult]:
        """
        按顺序执行多个技能（技能链）
        
        Args:
            skill_names: 技能名称列表
            initial_context: 初始上下文
            state: Agent状态
            
        Returns:
            执行结果列表
        """
        results = []
        context = initial_context.copy()
        
        for skill_name in skill_names:
            # 执行技能
            result = await self.execute_skill(skill_name, context, state)
            results.append(result)
            
            # 如果执行失败，中断链
            if not result.success:
                logger.warning(f"Skill chain interrupted at {skill_name}: {result.error}")
                break
            
            # 将结果传递给下一个技能
            if result.result:
                context["previous_result"] = result.result
        
        return results
    
    def recommend_next_skill(
        self,
        current_skill: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        根据当前技能和上下文推荐下一个技能
        
        Args:
            current_skill: 当前技能名称
            context: 当前上下文
            
        Returns:
            推荐的技能名称列表
        """
        recommendations = []
        
        # 获取当前技能的元数据
        current_metadata = self.skill_registry.get_skill_metadata(current_skill)
        if not current_metadata:
            return recommendations
        
        # 获取相同分类的其他技能
        same_category_skills = self.skill_registry.get_skills_by_category(
            current_metadata.category
        )
        
        # 过滤掉当前技能
        recommendations = [
            skill.name for skill in same_category_skills
            if skill.name != current_skill
        ]
        
        return recommendations[:3]  # 返回前3个推荐

    # ===== Worker相关方法 =====

    def parse_worker_instructions(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中解析Worker指令

        指令格式：[[EXECUTE_WORKER: @Worker_Name | INPUT: <Context>]]

        Args:
            text: 包含Worker指令的文本

        Returns:
            解析出的指令列表，每个指令包含worker_name和input_context
        """
        instructions = []

        # 正则表达式匹配Worker指令
        pattern = r'\[\[EXECUTE_WORKER:\s*(@\w+)\s*\|\s*INPUT:\s*([^\]]+)\]\]'
        matches = re.finditer(pattern, text, re.MULTILINE)

        for match in matches:
            worker_name = match.group(1).strip()  # @Douyin_Expert
            input_context = match.group(2).strip()  # 主题=xxx，风格=xxx

            # 解析INPUT参数
            input_params = self._parse_input_params(input_context)

            instruction = {
                "worker_name": worker_name,
                "input_context": input_context,
                "input_params": input_params
            }
            instructions.append(instruction)

            logger.debug(f"Parsed worker instruction: {worker_name} with params: {input_params}")

        return instructions

    def _parse_input_params(self, input_context: str) -> Dict[str, str]:
        """
        解析INPUT参数字符串

        格式：主题=AI技术，风格=科普，时长=30秒

        Args:
            input_context: INPUT参数字符串

        Returns:
            参数字典
        """
        params = {}

        # 按逗号分割
        parts = input_context.split(',')

        for part in parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                params[key.strip()] = value.strip()

        return params

    def load_worker_detail(self, worker_name: str, skill_name: str = "content-generate") -> Optional[str]:
        """
        加载Worker的详细指令文件

        Worker名称到文件的映射：
        - @Douyin_Expert → workers/douyin_worker.md
        - @Xiaohongshu_Expert → workers/xiaohongshu_worker.md
        - @Weibo_Expert → workers/weibo_worker.md

        Args:
            worker_name: Worker名称（如 @Douyin_Expert）
            skill_name: 技能名称（默认为 content-generate）

        Returns:
            Worker的详细指令内容，如果加载失败则返回None
        """
        # Worker名称到文件名的映射
        worker_file_map = {
            "@Douyin_Expert": "douyin_worker.md",
            "@Xiaohongshu_Expert": "xiaohongshu_worker.md",
            "@Weibo_Expert": "weibo_worker.md"
        }

        if worker_name not in worker_file_map:
            logger.error(f"Unknown worker name: {worker_name}")
            return None

        # 构建文件路径
        worker_file = worker_file_map[worker_name]
        worker_path = os.path.join(
            "meta_agent", "skills", skill_name, "workers", worker_file
        )

        try:
            # 读取Worker指令文件
            with open(worker_path, 'r', encoding='utf-8') as f:
                worker_detail = f.read()

            logger.debug(f"Loaded worker detail: {worker_name} from {worker_path}")
            return worker_detail

        except FileNotFoundError:
            logger.error(f"Worker file not found: {worker_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load worker detail {worker_name}: {e}")
            return None


    async def execute_worker(
        self,
        worker_name: str,
        input_params: Dict[str, str],
        skill_name: str = "content-generate",
        validate: bool = False,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        执行单个Worker（统一方法）
        
        Args:
            worker_name: Worker名称（如 @Douyin_Expert）
            input_params: 输入参数字典
            skill_name: 技能名称
            validate: 是否进行格式校验
            max_retries: 最大重试次数（仅当 validate=True 时生效）
            
        Returns:
            Worker执行结果
        """
        logger.debug(f"Starting worker execution: {worker_name} (validate={validate})")
        
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            # 如果需要验证，先进行格式校验
            if validate:
                input_str = ", ".join([f"{k}={v}" for k, v in input_params.items()])
                tool_call_str = f"[[EXECUTE_WORKER: {worker_name} | INPUT: {input_str}]]"
                
                is_valid, error_msg = self.interceptor.validate_tool_call_format(tool_call_str)
                
                if not is_valid:
                    logger.warning(f"Invalid tool call format (attempt {retry_count + 1}/{max_retries}): {error_msg}")
                    last_error = error_msg
                    retry_count += 1
                    
                    if retry_count < max_retries:
                        continue
                    else:
                        return {
                            "worker_name": worker_name,
                            "success": False,
                            "error": f"Format validation failed after {max_retries} attempts: {last_error}",
                            "content": None
                        }
            
            # 使用拦截器加载 Worker 详细指令（带缓存）
            worker_detail = self._get_cached_worker_detail(worker_name, skill_name)
            
            if not worker_detail:
                return {
                    "worker_name": worker_name,
                    "success": False,
                    "error": f"Failed to load worker detail: {worker_name}",
                    "content": None
                }
            
            try:
                # 调用LLM生成内容
                from meta_agent.core.llm_manager import LLMManager
                llm = LLMManager()
                
                # 构建用户输入
                user_input = "请根据以下参数生成内容：\n\n"
                for key, value in input_params.items():
                    user_input += f"- {key}: {value}\n"
                user_input += "\n请严格按照输出模板生成内容。"
                
                # 调用LLM生成内容
                content = await llm.generate(
                    system_prompt=worker_detail,
                    user_message=user_input
                )
                
                result = {
                    "worker_name": worker_name,
                    "success": True,
                    "content": content,
                    "metadata": {
                        "input_params": input_params,
                        "worker_detail_loaded": True,
                        "llm_called": True,
                        "used_cache": True,
                        "retry_count": retry_count
                    }
                }
                
                logger.debug(f"Worker executed successfully: {worker_name}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to execute worker {worker_name}: {e}")
                last_error = str(e)
                
                # 如果不需要重试或已达到最大重试次数，直接返回错误
                if not validate or retry_count >= max_retries - 1:
                    return {
                        "worker_name": worker_name,
                        "success": False,
                        "error": last_error,
                        "content": None
                    }
                
                # 否则重试
                retry_count += 1
                logger.warning(f"Worker execution failed (attempt {retry_count}/{max_retries}): {last_error}")
        
        # 达到最大重试次数
        return {
            "worker_name": worker_name,
            "success": False,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
            "content": None,
            "metadata": {
                "retry_count": retry_count
            }
        }

    async def execute_worker_with_validation(
        self,
        worker_name: str,
        input_params: Dict[str, str],
        skill_name: str = "content-generate",
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        执行 Worker 并进行格式校验和重试（兼容方法）
        
        此方法保留用于向后兼容，内部调用统一的 execute_worker 方法
        
        Args:
            worker_name: Worker名称
            input_params: 输入参数
            skill_name: 技能名称
            max_retries: 最大重试次数
            
        Returns:
            执行结果
        """
        return await self.execute_worker(
            worker_name=worker_name,
            input_params=input_params,
            skill_name=skill_name,
            validate=True,
            max_retries=max_retries
        )

    def _get_cached_worker_detail(self, worker_name: str, skill_name: str) -> Optional[str]:
        """
        获取缓存的 Worker 详细指令（LRU缓存）

        Args:
            worker_name: Worker名称
            skill_name: 技能名称

        Returns:
            Worker详细指令
        """
        cache_key = f"{skill_name}:{worker_name}"

        # 检查缓存
        if cache_key in self._worker_detail_cache:
            # 更新LRU顺序
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            logger.debug(f"Worker detail cache hit: {cache_key}")
            return self._worker_detail_cache[cache_key]

        # 缓存未命中，使用拦截器加载 Detail 层
        worker_detail = self.interceptor.load_worker_detail(worker_name, skill_name)

        if worker_detail:
            # 添加到缓存
            self._cache_worker_detail(cache_key, worker_detail)
            logger.debug(f"Worker detail loaded and cached: {cache_key}")

        return worker_detail

    def _cache_worker_detail(self, cache_key: str, worker_detail: str):
        """
        缓存 Worker 详细指令（LRU策略）

        Args:
            cache_key: 缓存键
            worker_detail: Worker详细指令
        """
        # 如果缓存已满，移除最旧的项
        if len(self._worker_detail_cache) >= self._max_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._worker_detail_cache[oldest_key]
            logger.debug(f"Cache full, removed oldest: {oldest_key}")

        # 添加新项
        self._worker_detail_cache[cache_key] = worker_detail
        self._cache_order.append(cache_key)

    def clear_worker_cache(self):
        """清空 Worker 指令缓存"""
        self._worker_detail_cache.clear()
        self._cache_order.clear()
        logger.debug("Worker detail cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            缓存统计字典
        """
        return {
            "cache_size": len(self._worker_detail_cache),
            "max_cache_size": self._max_cache_size,
            "cached_workers": list(self._worker_detail_cache.keys()),
            "interceptor_stats": self.interceptor.get_cache_stats()
        }

    async def execute_single_worker(
        self,
        instruction: Dict[str, Any],
        skill_name: str = "content-generate"
    ) -> Dict[str, Any]:
        """
        执行单个 Worker（Plan-Execute 模式使用）
        
        此方法保留用于向后兼容，内部调用统一的 execute_worker 方法

        Args:
            instruction: Worker指令，包含 worker_name 和 input_context
            skill_name: 技能名称

        Returns:
            Worker的执行结果
        """
        logger.debug(f"Executing single worker: {instruction.get('worker_name')}")

        try:
            worker_name = instruction["worker_name"]
            input_context = instruction["input_context"]

            # 解析输入参数
            input_params = self._parse_input_params(input_context)

            # 调用统一的 execute_worker 方法
            return await self.execute_worker(
                worker_name=worker_name,
                input_params=input_params,
                skill_name=skill_name,
                validate=False,
                max_retries=1
            )

        except Exception as e:
            logger.error(f"Error executing single worker: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "worker_name": instruction.get("worker_name", "unknown")
            }

    async def execute_workers(
        self,
        instructions: List[Dict[str, Any]],
        skill_name: str = "content-generate"
    ) -> Dict[str, Any]:
        """
        批量执行多个Worker

        Args:
            instructions: Worker指令列表
            skill_name: 技能名称

        Returns:
            所有Worker的执行结果
        """
        logger.debug(f"Starting batch worker execution: {len(instructions)} workers")
        results = {}

        for instruction in instructions:
            worker_name = instruction["worker_name"]
            input_params = instruction["input_params"]

            # 执行Worker
            result = await self.execute_worker(worker_name, input_params, skill_name)
            results[worker_name] = result

        # 统计执行情况
        total = len(instructions)
        successful = sum(1 for r in results.values() if r["success"])
        failed = total - successful

        summary = {
            "total_workers": total,
            "successful": successful,
            "failed": failed,
            "results": results
        }

        logger.debug(f"Workers execution completed: {successful}/{total} successful")
        return summary
