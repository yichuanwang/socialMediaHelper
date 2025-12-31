"""

工具定义 - 具体工具的Schema定义
只保留网页搜索和Shell命令两个工具
"""
from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod
import logging
from .tool_schemas import (
    ToolSchema,
    ParameterSchema,
    ReturnSchema,
    ToolExample,
    ToolCategory,
    ParameterType
)

logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """
    工具基类
    所有工具都应继承此类并实现execute方法
    """
    
    def __init__(self, schema: ToolSchema):
        """
        初始化工具
        
        Args:
            schema: 工具的Schema定义
        """
        self.schema = schema
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        pass
    
    def validate_and_execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证输入并执行工具
        
        Args:
            input_data: 输入参数
            
        Returns:
            包含结果或错误信息的字典
        """
        # 验证输入
        is_valid, error_msg = self.schema.validate_input(input_data)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg
            }
        
        # 执行工具（同步包装）
        try:
            import asyncio
            result = asyncio.run(self.execute(**input_data))
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# ===== 工具定义 =====

# 1. 网络搜索工具
SEARCH_WEB_SCHEMA = ToolSchema(
    name="search_web",
    display_name="网络搜索",
    description="在互联网上搜索信息，返回相关的搜索结果",
    category=ToolCategory.WEB_INTERACTION,
    parameters=[
        ParameterSchema(
            name="query",
            type=ParameterType.STRING,
            description="搜索查询词，应该清晰、具体",
            required=True,
            min_length=1,
            max_length=500,
            example="Python异步编程最佳实践"
        ),
        ParameterSchema(
            name="max_results",
            type=ParameterType.INTEGER,
            description="返回的最大结果数量",
            required=False,
            default=10,
            min_value=1,
            max_value=50,
            example=10
        ),
        ParameterSchema(
            name="language",
            type=ParameterType.STRING,
            description="搜索结果的语言",
            required=False,
            default="zh-CN",
            enum=["zh-CN", "en-US", "ja-JP"],
            example="zh-CN"
        )
    ],
    returns=ReturnSchema(
        type=ParameterType.ARRAY,
        description="搜索结果列表，每个结果包含标题、URL、摘要",
        example=[
            {
                "title": "Python异步编程指南",
                "url": "https://example.com/async",
                "snippet": "详细介绍Python的asyncio库..."
            }
        ]
    ),
    examples=[
        ToolExample(
            description="搜索Python相关技术文章",
            input={"query": "Python异步编程", "max_results": 5},
            output=[{"title": "...", "url": "...", "snippet": "..."}],
            explanation="返回前5个最相关的搜索结果"
        )
    ],
    use_cases=[
        "查找技术文档和教程",
        "研究特定主题的最新信息",
        "收集多个来源的观点"
    ],
    notes=[
        "搜索结果的质量取决于查询词的准确性",
        "建议使用具体的关键词而非宽泛的描述"
    ],
    estimated_time="1-3秒",
    tags=["搜索", "网络", "信息检索"]
)

# 2. 执行Shell命令工具
EXECUTE_SHELL_SCHEMA = ToolSchema(
    name="execute_shell",
    display_name="执行Shell命令",
    description="在系统shell中执行命令",
    category=ToolCategory.UTILITY,
    parameters=[
        ParameterSchema(
            name="command",
            type=ParameterType.STRING,
            description="要执行的shell命令",
            required=True,
            example="ls -la"
        ),
        ParameterSchema(
            name="working_directory",
            type=ParameterType.STRING,
            description="工作目录",
            required=False,
            example="/home/user/project"
        ),
        ParameterSchema(
            name="timeout",
            type=ParameterType.INTEGER,
            description="超时时间（秒）",
            required=False,
            default=60,
            min_value=1,
            max_value=600,
            example=60
        )
    ],
    returns=ReturnSchema(
        type=ParameterType.OBJECT,
        description="命令执行结果，包含返回码、标准输出和标准错误",
        example={
            "return_code": 0,
            "stdout": "file1.txt\nfile2.txt",
            "stderr": ""
        }
    ),
    examples=[
        ToolExample(
            description="列出目录内容",
            input={"command": "ls -l"},
            output={"return_code": 0, "stdout": "total 8\n-rw-r--r-- 1 user..."},
            explanation="成功执行ls命令"
        )
    ],
    use_cases=[
        "自动化脚本执行",
        "系统管理任务",
        "构建和部署"
    ],
    notes=[
        "谨慎使用，避免执行危险命令",
        "需要适当的系统权限"
    ],
    prerequisites=[
        "有执行命令的权限",
        "命令在系统PATH中可用"
    ],
    estimated_time="1-60秒",
    tags=["shell", "命令", "系统"]
)

# 工具Schema字典（方便查找）
TOOL_SCHEMAS = {
    "search_web": SEARCH_WEB_SCHEMA,
    "execute_shell": EXECUTE_SHELL_SCHEMA,
}

def get_tool_schema(tool_name: str) -> Optional[ToolSchema]:
    """
    根据工具名称获取Schema
    
    Args:
        tool_name: 工具名称
        
    Returns:
        工具Schema，如果不存在返回None
    """
    return TOOL_SCHEMAS.get(tool_name)

def get_all_tool_names() -> list[str]:
    """
    获取所有工具名称列表
    
    Returns:
        工具名称列表
    """
    return list(TOOL_SCHEMAS.keys())

def get_tools_by_category(category: ToolCategory) -> list[ToolSchema]:
    """
    根据分类获取工具列表
    
    Args:
        category: 工具分类
        
    Returns:
        该分类下的所有工具Schema
    """
    return [
        schema for schema in TOOL_SCHEMAS.values()
        if schema.category == category
    ]

# ===== 工具实现类 =====

class SearchWebTool(BaseTool):
    """网络搜索工具实现"""

    def __init__(self):
        super().__init__(SEARCH_WEB_SCHEMA)
        self._ddgs = None

    def _get_ddgs(self):
        """延迟初始化 DuckDuckGo 搜索客户端"""
        if self._ddgs is None:
            try:
                from ddgs import DDGS
                self._ddgs = DDGS()
            except ImportError:
                raise ImportError(
                    "ddgs is not installed. "
                    "Please install it with: pip install ddgs"
                )
        return self._ddgs

    async def execute(
        self,
        query: str,
        max_results: int = 10,
        language: str = "zh-CN",
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        执行网络搜索

        Args:
            query: 搜索查询词
            max_results: 最大结果数量
            language: 搜索语言

        Returns:
            搜索结果列表
        """
        try:
            ddgs = self._get_ddgs()

            # 设置地区参数
            region = "cn-zh" if language == "zh-CN" else "us-en"

            # 执行搜索
            results = []
            try:
                search_results = ddgs.text(
                    query=query,
                    region=region,
                    max_results=max_results
                )

                # 检查搜索结果
                if search_results is None:
                    logger.warning(f"Search returned None for query: {query}")
                    return [{
                        "title": "搜索暂时不可用",
                        "url": "",
                        "snippet": f"抱歉，当前无法搜索 '{query}'。这可能是由于网络问题或搜索服务限制。请稍后再试或换个关键词。"
                    }]

                # 格式化结果
                for result in search_results:
                    if result:  # 确保结果不为 None
                        results.append({
                            "title": result.get("title", "无标题"),
                            "url": result.get("href", ""),
                            "snippet": result.get("body", "无摘要")
                        })

                # 如果没有结果
                if not results:
                    logger.debug(f"No results found for query: {query}")
                    return [{
                        "title": "未找到相关结果",
                        "url": "",
                        "snippet": f"没有找到关于 '{query}' 的相关信息。建议尝试使用不同的关键词或更具体的搜索词。"
                    }]

                logger.debug(f"Search completed: query='{query}', results={len(results)}")
                return results

            except Exception as search_error:
                logger.error(f"DuckDuckGo search error: {search_error}")
                # 返回友好的错误信息而不是抛出异常
                return [{
                    "title": "搜索遇到问题",
                    "url": "",
                    "snippet": f"搜索 '{query}' 时遇到问题: {str(search_error)}。这可能是暂时的网络问题，请稍后重试。"
                }]

        except ImportError as ie:
            logger.error(f"Import error: {ie}")
            raise Exception(f"搜索工具未正确安装: {str(ie)}")
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            raise Exception(f"搜索失败: {str(e)}")


class ExecuteShellTool(BaseTool):
    """Shell命令执行工具实现"""

    def __init__(self):
        super().__init__(EXECUTE_SHELL_SCHEMA)

    async def execute(
        self,
        command: str,
        working_directory: Optional[str] = None,
        timeout: int = 60,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行Shell命令

        Args:
            command: 要执行的命令
            working_directory: 工作目录
            timeout: 超时时间（秒）

        Returns:
            执行结果字典
        """
        import asyncio
        import os

        try:
            # 设置工作目录
            cwd = working_directory if working_directory else os.getcwd()

            # 执行命令
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )

            # 等待命令完成（带超时）
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise Exception(f"命令执行超时（{timeout}秒）")

            # 返回结果
            result = {
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore')
            }

            logger.debug(f"Command executed: {command}, return_code={result['return_code']}")
            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise Exception(f"命令执行失败: {str(e)}")


# 工具实例
SEARCH_WEB_TOOL = SearchWebTool()
EXECUTE_SHELL_TOOL = ExecuteShellTool()
