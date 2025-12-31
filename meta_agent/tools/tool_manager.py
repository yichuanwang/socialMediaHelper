"""
工具管理器 - 管理工具的注册、查询和调用
"""
from typing import Dict, List, Optional, Any, Callable
import logging
from .tool_schemas import ToolSchema, ToolCategory
from .tool_definitions import (
    TOOL_SCHEMAS,
    get_tool_schema,
    get_all_tool_names,
    get_tools_by_category,
    SEARCH_WEB_TOOL,
    EXECUTE_SHELL_TOOL
)


logger = logging.getLogger(__name__)


class ToolManager:
    """
    工具管理器
    负责工具的注册、查询、验证和调用管理
    """
    
    def __init__(self):
        """初始化工具管理器"""
        # 工具Schema注册表
        self._tool_schemas: Dict[str, ToolSchema] = {}
        
        # 工具实现注册表（工具名 -> 执行函数）
        self._tool_implementations: Dict[str, Callable] = {}
        
        # 工具分类索引
        self._category_index: Dict[ToolCategory, List[str]] = {}
        
        # 工具标签索引
        self._tag_index: Dict[str, List[str]] = {}
        
        # 加载预定义的工具Schema
        self._load_predefined_tools()
        
        logger.debug(f"ToolManager initialized with {len(self._tool_schemas)} tools")
    
    def _load_predefined_tools(self):
        """加载预定义的工具Schema和实现"""
        for tool_name, schema in TOOL_SCHEMAS.items():
            self.register_tool_schema(schema)

        # 注册工具实现
        self.register_tool_implementation("search_web", SEARCH_WEB_TOOL.execute)
        self.register_tool_implementation("execute_shell", EXECUTE_SHELL_TOOL.execute)
    
    def register_tool_schema(self, schema: ToolSchema) -> bool:
        """
        注册工具Schema
        
        Args:
            schema: 工具Schema
            
        Returns:
            是否注册成功
        """
        try:
            tool_name = schema.name
            
            # 检查是否已存在
            if tool_name in self._tool_schemas:
                logger.warning(f"Tool {tool_name} already registered, will be overwritten")
            
            # 注册Schema
            self._tool_schemas[tool_name] = schema
            
            # 更新分类索引
            if schema.category not in self._category_index:
                self._category_index[schema.category] = []
            if tool_name not in self._category_index[schema.category]:
                self._category_index[schema.category].append(tool_name)
            
            # 更新标签索引
            for tag in schema.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = []
                if tool_name not in self._tag_index[tag]:
                    self._tag_index[tag].append(tool_name)
            
            logger.debug(f"Tool schema registered: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool schema: {e}")
            return False
    
    def register_tool_implementation(
        self,
        tool_name: str,
        implementation: Callable
    ) -> bool:
        """
        注册工具实现
        
        Args:
            tool_name: 工具名称
            implementation: 工具实现函数
            
        Returns:
            是否注册成功
        """
        try:
            # 检查Schema是否存在
            if tool_name not in self._tool_schemas:
                logger.error(f"Tool schema not found: {tool_name}")
                return False
            
            # 注册实现
            self._tool_implementations[tool_name] = implementation
            logger.debug(f"Tool implementation registered: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool implementation: {e}")
            return False
    
    def get_tool_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """
        获取工具Schema
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具Schema，如果不存在返回None
        """
        return self._tool_schemas.get(tool_name)
    
    def get_all_tool_names(self) -> List[str]:
        """
        获取所有工具名称
        
        Returns:
            工具名称列表
        """
        return list(self._tool_schemas.keys())
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolSchema]:
        """
        根据分类获取工具
        
        Args:
            category: 工具分类
            
        Returns:
            该分类下的所有工具Schema
        """
        tool_names = self._category_index.get(category, [])
        return [self._tool_schemas[name] for name in tool_names]
    
    def get_tools_by_tag(self, tag: str) -> List[ToolSchema]:
        """
        根据标签获取工具
        
        Args:
            tag: 标签
            
        Returns:
            包含该标签的所有工具Schema
        """
        tool_names = self._tag_index.get(tag, [])
        return [self._tool_schemas[name] for name in tool_names]
    
    def search_tools(
        self,
        query: str,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None
    ) -> List[ToolSchema]:
        """
        搜索工具
        
        Args:
            query: 搜索关键词
            category: 工具分类过滤
            tags: 标签过滤
            
        Returns:
            匹配的工具Schema列表
        """
        results = []
        query_lower = query.lower()
        
        for tool_name, schema in self._tool_schemas.items():
            # 分类过滤
            if category and schema.category != category:
                continue
            
            # 标签过滤
            if tags and not any(tag in schema.tags for tag in tags):
                continue
            
            # 关键词匹配
            if (query_lower in schema.name.lower() or
                query_lower in schema.display_name.lower() or
                query_lower in schema.description.lower() or
                any(query_lower in uc.lower() for uc in schema.use_cases)):
                results.append(schema)
        
        return results
    
    def validate_tool_input(
        self,
        tool_name: str,
        input_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        验证工具输入
        
        Args:
            tool_name: 工具名称
            input_data: 输入数据
            
        Returns:
            (是否有效, 错误信息)
        """
        schema = self.get_tool_schema(tool_name)
        if not schema:
            return False, f"Tool not found: {tool_name}"
        
        return schema.validate_input(input_data)
    
    def get_tool_for_llm(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        获取适合LLM使用的工具描述
        
        Args:
            tool_name: 工具名称
            
        Returns:
            LLM格式的工具描述
        """
        schema = self.get_tool_schema(tool_name)
        if not schema:
            return None
        
        return schema.to_llm_format()
    
    def get_all_tools_for_llm(
        self,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        获取所有工具的LLM格式描述
        
        Args:
            category: 可选的分类过滤
            tags: 可选的标签过滤
            
        Returns:
            LLM格式的工具描述列表
        """
        tools = []
        
        for tool_name, schema in self._tool_schemas.items():
            # 分类过滤
            if category and schema.category != category:
                continue
            
            # 标签过滤
            if tags and not any(tag in schema.tags for tag in tags):
                continue
            
            tools.append(schema.to_llm_format())
        
        return tools
    
    def has_implementation(self, tool_name: str) -> bool:
        """
        检查工具是否有实现
        
        Args:
            tool_name: 工具名称
            
        Returns:
            是否有实现
        """
        return tool_name in self._tool_implementations
    
    async def execute_tool(
        self,
        tool_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            input_data: 输入数据
            
        Returns:
            执行结果字典，包含success、result或error字段
        """
        try:
            # 验证输入
            is_valid, error_msg = self.validate_tool_input(tool_name, input_data)
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # 检查实现
            if not self.has_implementation(tool_name):
                return {
                    "success": False,
                    "error": f"Tool implementation not found: {tool_name}"
                }
            
            # 执行工具
            implementation = self._tool_implementations[tool_name]
            result = await implementation(**input_data)
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """
        获取工具统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_tools": len(self._tool_schemas),
            "implemented_tools": len(self._tool_implementations),
            "categories": {
                category.value: len(tools)
                for category, tools in self._category_index.items()
            },
            "total_tags": len(self._tag_index)
        }
    
    def get_recommended_tools(
        self,
        task_description: str,
        max_results: int = 5
    ) -> List[ToolSchema]:
        """
        根据任务描述推荐工具
        
        Args:
            task_description: 任务描述
            max_results: 最大返回数量
            
        Returns:
            推荐的工具Schema列表
        """
        # 简单的关键词匹配推荐
        # 实际应用中可以使用更复杂的语义匹配
        task_lower = task_description.lower()
        scored_tools = []
        
        for tool_name, schema in self._tool_schemas.items():
            score = 0
            
            # 名称匹配
            if any(word in schema.name.lower() for word in task_lower.split()):
                score += 3
            
            # 描述匹配
            if any(word in schema.description.lower() for word in task_lower.split()):
                score += 2
            
            # 使用场景匹配
            for use_case in schema.use_cases:
                if any(word in use_case.lower() for word in task_lower.split()):
                    score += 1
            
            if score > 0:
                scored_tools.append((score, schema))
        
        # 按分数排序
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        
        # 返回前N个
        return [schema for _, schema in scored_tools[:max_results]]
    
    def export_tool_catalog(self) -> Dict[str, Any]:
        """
        导出工具目录
        
        Returns:
            工具目录字典，包含所有工具的详细信息
        """
        catalog = {
            "version": "1.0.0",
            "total_tools": len(self._tool_schemas),
            "categories": {},
            "tools": {}
        }
        
        # 按分类组织
        for category in ToolCategory:
            tools_in_category = self.get_tools_by_category(category)
            if tools_in_category:
                catalog["categories"][category.value] = [
                    tool.name for tool in tools_in_category
                ]
        
        # 添加工具详情
        for tool_name, schema in self._tool_schemas.items():
            catalog["tools"][tool_name] = {
                "display_name": schema.display_name,
                "description": schema.description,
                "category": schema.category.value,
                "version": schema.version,
                "parameters_count": len(schema.parameters),
                "has_implementation": self.has_implementation(tool_name),
                "tags": schema.tags,
                "use_cases": schema.use_cases
            }
        
        return catalog


# 全局工具管理器实例
_global_tool_manager: Optional[ToolManager] = None


def get_tool_manager() -> ToolManager:
    """
    获取全局工具管理器实例（单例模式）
    
    Returns:
        工具管理器实例
    """
    global _global_tool_manager
    if _global_tool_manager is None:
        _global_tool_manager = ToolManager()
    return _global_tool_manager


def reset_tool_manager():
    """重置全局工具管理器（主要用于测试）"""
    global _global_tool_manager
    _global_tool_manager = None
