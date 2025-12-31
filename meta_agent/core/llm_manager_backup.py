"""
LLM管理器 - 统一管理LLM调用
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class LLMManager:
    """
    LLM管理器
    负责统一管理和调用LLM
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None
    ):
        """
        初始化LLM管理器
        
        Args:
            model: 模型名称，默认从环境变量读取
            temperature: 温度参数，默认从环境变量读取
            api_key: API密钥，默认从环境变量读取
        """
        # 加载环境变量
        load_dotenv()
        
        # 获取配置
        self.model_name = model or os.getenv("DEFAULT_MODEL", "gpt-4")
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.7"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # 验证API密钥
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        
        # 初始化模型
        self.model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )
        
        logger.debug(f"LLMManager initialized with model: {self.model_name}")
    
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成回复
        
        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            temperature: 温度参数（可选，覆盖默认值）
            max_tokens: 最大token数（可选）
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        try:
            # 构建消息
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            # 准备调用参数
            invoke_kwargs = kwargs.copy()
            if temperature is not None:
                invoke_kwargs["temperature"] = temperature
            if max_tokens is not None:
                invoke_kwargs["max_tokens"] = max_tokens
            
            # 调用LLM
            logger.debug(f"Calling LLM with model: {self.model_name}")
            logger.debug(f"System prompt length: {len(system_prompt)}")
            logger.debug(f"User message length: {len(user_message)}")

            response = await self.model.ainvoke(messages, **invoke_kwargs)

            logger.debug("LLM response received")
            logger.debug(f"Response length: {len(response.content)}")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    async def generate_with_history(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        带历史记录的生成
        
        Args:
            system_prompt: 系统提示词
            messages: 消息历史，格式为 [{"role": "user/assistant", "content": "..."}]
            temperature: 温度参数（可选）
            max_tokens: 最大token数（可选）
            **kwargs: 其他参数
            
        Returns:
            生成的文本内容
        """
        try:
            # 构建消息列表
            message_list = [SystemMessage(content=system_prompt)]
            
            for msg in messages:
                if msg["role"] == "user":
                    message_list.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    from langchain_core.messages import AIMessage
                    message_list.append(AIMessage(content=msg["content"]))
            
            # 准备调用参数
            invoke_kwargs = kwargs.copy()
            if temperature is not None:
                invoke_kwargs["temperature"] = temperature
            if max_tokens is not None:
                invoke_kwargs["max_tokens"] = max_tokens
            
            # 调用LLM
            logger.debug(f"Calling LLM with {len(messages)} history messages")
            response = await self.model.ainvoke(message_list, **invoke_kwargs)

            logger.debug("LLM response received")
            return response.content
            
        except Exception as e:
            logger.error(f"Error calling LLM with history: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "api_key_configured": bool(self.api_key)
        }

    async def generate_with_interception(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        interceptor: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        带拦截机制的生成

        支持在生成过程中检测工具调用标记，自动加载 Summary 并注入上下文

        Args:
            system_prompt: 系统提示词
            messages: 消息历史
            interceptor: SkillInterceptor 实例
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            生成的文本内容
        """
        from langchain_core.messages import AIMessage

        try:
            # 如果没有提供拦截器，使用普通生成
            if interceptor is None:
                return await self.generate_with_history(
                    system_prompt, messages, temperature, max_tokens, **kwargs
                )

            # 构建消息列表
            message_list = [SystemMessage(content=system_prompt)]

            for msg in messages:
                if msg["role"] == "user":
                    message_list.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    message_list.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "developer":
                    # Developer message 用于注入 Summary
                    from langchain_core.messages import SystemMessage as DevMessage
                    message_list.append(DevMessage(content=msg["content"]))

            # 准备调用参数
            invoke_kwargs = kwargs.copy()
            if temperature is not None:
                invoke_kwargs["temperature"] = temperature
            if max_tokens is not None:
                invoke_kwargs["max_tokens"] = max_tokens

            # 第一次调用 LLM
            logger.debug("First LLM call with interception enabled")
            response = await self.model.ainvoke(message_list, **invoke_kwargs)
            response_text = response.content

            # 检测工具调用
            tool_call_info = interceptor.intercept_tool_call(response_text)

            if tool_call_info is None:
                # 没有检测到工具调用，直接返回
                logger.debug("No tool call detected, returning response")
                return response_text

            logger.debug(f"Tool call detected: {tool_call_info.tool_name}")

            # 验证格式
            if not tool_call_info.is_valid:
                # 格式错误，返回错误反馈
                error_feedback = interceptor.generate_error_feedback(tool_call_info)
                logger.warning(f"Invalid tool call format: {tool_call_info.error_message}")
                return error_feedback

            # 加载 Worker Detail 层
            detail = interceptor.load_worker_detail(tool_call_info.tool_name)

            if detail is None:
                logger.error(f"Failed to load detail for {tool_call_info.tool_name}")
                return response_text

            logger.debug(f"Detail loaded for {tool_call_info.tool_name}, injecting into context")

            # 注入 Detail 到上下文
            # 在最后一个 user message 之后注入
            injected_messages = self._inject_detail_to_context(
                messages, detail, tool_call_info.tool_name
            )

            # 二次推理：使用注入了 Detail 的上下文重新生成
            logger.debug("Second LLM call with injected detail")
            final_response = await self.generate_with_history(
                system_prompt, injected_messages, temperature, max_tokens, **kwargs
            )

            return final_response

        except Exception as e:
            logger.error(f"Error in generate_with_interception: {e}")
            raise

    def _inject_detail_to_context(
        self,
        messages: List[Dict[str, str]],
        detail: str,
        tool_name: str
    ) -> List[Dict[str, str]]:
        """
        将 Worker Detail 层注入到消息上下文中

        Args:
            messages: 原始消息列表
            detail: Worker 的详细指令（Detail 层）
            tool_name: Worker 名称

        Returns:
            注入后的消息列表
        """
        from datetime import datetime

        # 找到最后一个 user message 的位置
        last_user_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_index = i
                break

        # 构建注入的消息
        injected_message = {
            "role": "developer",
            "content": f"# Worker 详细指令: {tool_name}\n\n{detail}",
            "metadata": {
                "type": "skill_summary",
                "lifecycle": "active",
                "tool_name": tool_name,
                "injected_at": datetime.now().isoformat()
            }
        }

        # 在最后一个 user message 之后插入
        new_messages = messages.copy()
        if last_user_index >= 0:
            new_messages.insert(last_user_index + 1, injected_message)
        else:
            # 如果没有 user message，追加到末尾
            new_messages.append(injected_message)

        logger.debug(f"Summary injected after message index {last_user_index}")
        return new_messages

    async def resume_generation(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        injected_content: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        恢复生成（二次推理）

        在注入新内容后继续生成

        Args:
            system_prompt: 系统提示词
            messages: 消息历史
            injected_content: 注入的内容
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            生成的文本内容
        """
        # 将注入的内容添加到消息历史
        enhanced_messages = messages.copy()
        enhanced_messages.append({
            "role": "developer",
            "content": injected_content
        })

        # 继续生成
        return await self.generate_with_history(
            system_prompt, enhanced_messages, temperature, max_tokens, **kwargs
        )

    def manage_context_window(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8000
    ) -> List[Dict[str, str]]:
        """
        管理上下文窗口大小

        当消息过多时进行压缩

        Args:
            messages: 消息列表
            max_tokens: 最大 token 数（粗略估计）

        Returns:
            压缩后的消息列表
        """
        # 粗略估计：每个字符约 0.5 个 token（中文）或 0.25 个 token（英文）
        # 这里使用保守估计：每个字符 0.5 个 token

        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = total_chars * 0.5

        if estimated_tokens <= max_tokens:
            return messages

        logger.debug(f"Context window exceeds limit ({estimated_tokens:.0f} > {max_tokens}), compressing...")

        # 压缩策略：保留最近的消息和重要的消息
        # 1. 始终保留最后 5 条消息
        # 2. 移除中间的 assistant 消息（保留 user 和 developer 消息）

        if len(messages) <= 5:
            return messages

        # 保留最后 5 条
        recent_messages = messages[-5:]

        # 从剩余消息中筛选重要消息
        older_messages = messages[:-5]
        important_messages = []

        for msg in older_messages:
            # 保留 user 和 developer 消息
            if msg["role"] in ["user", "developer"]:
                important_messages.append(msg)

        # 组合消息
        compressed_messages = important_messages + recent_messages

        logger.debug(f"Compressed from {len(messages)} to {len(compressed_messages)} messages")
        return compressed_messages
