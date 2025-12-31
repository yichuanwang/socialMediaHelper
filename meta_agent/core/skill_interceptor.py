"""
SkillInterceptor - 技能拦截器
实现工具调用的自动拦截和 Summary 动态加载机制
"""
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ToolCallInfo:
    """工具调用信息"""
    tool_name: str  # 如 @Douyin_Expert
    raw_call: str  # 原始调用字符串
    parameters: Dict[str, str]  # 解析出的参数
    is_valid: bool  # 格式是否有效
    start_pos: int = 0  # 匹配开始位置
    end_pos: int = 0  # 匹配结束位置
    error_message: Optional[str] = None  # 错误信息


class SkillInterceptor:
    """
    技能拦截器
    
    核心功能：
    1. 检测 LLM 输出中的工具调用标记
    2. 自动加载对应的 Worker Summary
    3. 验证工具调用格式
    4. 管理 Summary 缓存
    """
    
    # 工具调用格式正则表达式
    # 修复：使用更贪婪的匹配，匹配除了 ]] 之外的所有字符（包括中文、逗号等）
    TOOL_CALL_PATTERN = r'\[\[EXECUTE_WORKER:\s*(@\w+)\s*\|\s*INPUT:\s*([^\]]+)\]\]'
    
    def __init__(self, skills_dir: Path):
        """
        初始化拦截器
        
        Args:
            skills_dir: 技能目录路径
        """
        self.skills_dir = Path(skills_dir)
        self._summary_cache: Dict[str, str] = {}  # Worker名称 -> Summary内容
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.debug("SkillInterceptor initialized")
    
    def intercept_tool_call(self, response: str) -> Optional[ToolCallInfo]:
        """
        拦截并解析工具调用
        
        Args:
            response: LLM 的响应文本
            
        Returns:
            工具调用信息，如果没有检测到则返回 None
        """
        # 增强调试：记录响应长度和预览
        logger.debug(f"Searching for tool calls in response (length: {len(response)})")
        logger.debug(f"Response preview (first 500 chars): {response[:500]}...")

        # 使用正则表达式查找工具调用（添加 re.DOTALL 支持多行匹配）
        match = re.search(self.TOOL_CALL_PATTERN, response, re.DOTALL)

        if not match:
            logger.warning("No tool call pattern matched")
            logger.debug(f"Full response for debugging:\n{response}")
            return None

        # 提取工具名称和参数
        tool_name = match.group(1)  # @Douyin_Expert
        input_str = match.group(2)  # 主题=AI技术, 风格=科普
        raw_call = match.group(0)  # 完整的调用字符串

        logger.debug(f"Matched tool call - Name: {tool_name}, Input: {input_str[:100]}...")

        # 解析参数
        parameters = self._parse_input_params(input_str)
        logger.debug(f"Parsed parameters: {parameters}")
        
        # 验证格式
        is_valid, error_msg = self.validate_tool_call_format(raw_call)
        
        tool_call_info = ToolCallInfo(
            tool_name=tool_name,
            raw_call=raw_call,
            parameters=parameters,
            is_valid=is_valid,
            start_pos=match.start(),
            end_pos=match.end(),
            error_message=error_msg
        )

        logger.debug(f"Intercepted tool call: {tool_name}, valid: {is_valid}, pos: {match.start()}-{match.end()}")
        return tool_call_info
    
    def detect_tool_call_in_stream(self, text_chunk: str, buffer: str) -> Tuple[bool, str]:
        """
        在流式输出中检测工具调用
        
        Args:
            text_chunk: 当前接收到的文本块
            buffer: 累积的缓冲区
            
        Returns:
            (是否检测到完整的工具调用, 更新后的缓冲区)
        """
        # 更新缓冲区
        buffer += text_chunk
        
        # 检查是否包含完整的工具调用标记
        if '[[EXECUTE_WORKER:' in buffer and ']]' in buffer:
            # 尝试匹配完整的工具调用（添加 re.DOTALL）
            match = re.search(self.TOOL_CALL_PATTERN, buffer, re.DOTALL)
            if match:
                logger.debug("Complete tool call detected in stream")
                logger.debug(f"Matched content: {match.group(0)[:200]}...")
                return True, buffer
        
        # 保持缓冲区大小合理（最多保留最后 500 个字符）
        if len(buffer) > 500:
            buffer = buffer[-500:]
        
        return False, buffer
    
    def load_worker_detail(self, tool_name: str, skill_name: str = "content-generate") -> Optional[str]:
        """
        加载 Worker 的详细指令（Detail 层）

        Args:
            tool_name: Worker 名称，如 @Douyin_Expert
            skill_name: 技能名称

        Returns:
            Worker 的详细指令内容（Detail 层），如果加载失败则返回 None
        """
        # 检查缓存
        cache_key = f"{skill_name}:{tool_name}"
        if cache_key in self._summary_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for {tool_name}")
            return self._summary_cache[cache_key]

        self._cache_misses += 1

        # Worker 名称到文件名的映射
        worker_file_map = {
            "@Douyin_Expert": "douyin.md",
            "@Xiaohongshu_Expert": "xiaohongshu.md",
            "@Weibo_Expert": "weibo.md",
            "@Instagram_Expert": "instagram.md"
        }

        if tool_name not in worker_file_map:
            logger.error(f"Unknown worker: {tool_name}")
            return None

        # 构建文件路径
        worker_file = worker_file_map[tool_name]
        worker_path = self.skills_dir / skill_name / "platforms" / worker_file

        try:
            # 读取 Worker 详细指令（Detail 层）
            detail = worker_path.read_text(encoding='utf-8')

            # 缓存结果
            self._summary_cache[cache_key] = detail

            logger.debug(f"Loaded worker detail: {tool_name} from {worker_path}")
            return detail

        except FileNotFoundError:
            logger.error(f"Worker file not found: {worker_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load worker detail {tool_name}: {e}")
            return None
    
    def validate_tool_call_format(self, tool_call: str) -> Tuple[bool, Optional[str]]:
        """
        验证工具调用格式
        
        Args:
            tool_call: 工具调用字符串
            
        Returns:
            (是否有效, 错误信息)
        """
        logger.debug(f"Validating tool call format: {tool_call[:100]}...")

        # 检查基本格式
        if not tool_call.startswith('[[EXECUTE_WORKER:'):
            logger.debug("Validation failed: doesn't start with [[EXECUTE_WORKER:")
            return False, "Tool call must start with '[[EXECUTE_WORKER:'"

        if not tool_call.endswith(']]'):
            logger.debug("Validation failed: doesn't end with ]]")
            return False, "Tool call must end with ']]'"

        # 检查是否包含 Worker 名称（以 @ 开头）
        if '@' not in tool_call:
            logger.debug("Validation failed: no @ symbol found")
            return False, "Worker name must start with '@'"

        # 检查是否包含分隔符
        if '|' not in tool_call:
            logger.debug("Validation failed: no | separator found")
            return False, "Must use '|' to separate worker name and parameters"

        # 检查是否包含 INPUT 标记
        if 'INPUT:' not in tool_call:
            logger.debug("Validation failed: no INPUT: keyword found")
            return False, "Must include 'INPUT:' keyword"

        # 使用正则表达式完整验证（添加 re.DOTALL）
        match = re.match(self.TOOL_CALL_PATTERN, tool_call, re.DOTALL)
        if not match:
            logger.debug("Validation failed: regex pattern didn't match")
            return False, "Invalid tool call format. Please use: [[EXECUTE_WORKER: @WorkerName | INPUT: param1=value1, param2=value2]]"

        logger.debug("Validation passed")
        return True, None
    
    def generate_error_feedback(self, tool_call_info: ToolCallInfo) -> str:
        """
        生成格式错误的反馈信息
        
        Args:
            tool_call_info: 工具调用信息
            
        Returns:
            错误反馈字符串
        """
        feedback = f"""Error: Invalid tool call format detected.

Your input: {tool_call_info.raw_call}

Error: {tool_call_info.error_message}

Correct format:
[[EXECUTE_WORKER: @WorkerName | INPUT: param1=value1, param2=value2]]

Examples:
[[EXECUTE_WORKER: @Douyin_Expert | INPUT: 主题=AI技术, 风格=科普, 时长=30秒]]
[[EXECUTE_WORKER: @Xiaohongshu_Expert | INPUT: 主题=旅行攻略, 风格=种草]]

Please correct your tool call and try again."""
        
        return feedback
    
    def _parse_input_params(self, input_str: str) -> Dict[str, str]:
        """
        解析 INPUT 参数字符串
        
        Args:
            input_str: INPUT 参数字符串，如 "主题=AI技术, 风格=科普" 或 "主题=AI技术，风格=科普"

        Returns:
            参数字典
        """
        params = {}

        # 按逗号分割（支持中英文逗号）
        parts = re.split(r'[,，]', input_str)
        
        for part in parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                params[key.strip()] = value.strip()
        
        return params
    
    def clear_cache(self):
        """清空 Summary 缓存"""
        self._summary_cache.clear()
        logger.debug("Summary cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._summary_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.2%}",
            "cached_workers": list(self._summary_cache.keys())
        }
