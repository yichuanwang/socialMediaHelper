"""
工具Schema定义 - 清晰、结构化、无歧义的工具描述
使用Pydantic模型确保类型安全和验证
"""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ParameterType(str, Enum):
    """参数类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ParameterSchema(BaseModel):
    """工具参数Schema"""
    name: str = Field(description="参数名称")
    type: ParameterType = Field(description="参数类型")
    description: str = Field(description="参数的详细描述")
    required: bool = Field(default=False, description="是否必填")
    default: Optional[Any] = Field(default=None, description="默认值")
    
    # 约束条件
    min_value: Optional[float] = Field(default=None, description="最小值（数值类型）")
    max_value: Optional[float] = Field(default=None, description="最大值（数值类型）")
    min_length: Optional[int] = Field(default=None, description="最小长度（字符串/数组）")
    max_length: Optional[int] = Field(default=None, description="最大长度（字符串/数组）")
    pattern: Optional[str] = Field(default=None, description="正则表达式模式（字符串）")
    enum: Optional[List[Any]] = Field(default=None, description="枚举值列表")
    
    # 示例
    example: Optional[Any] = Field(default=None, description="参数示例值")
    
    class Config:
        use_enum_values = True


class ReturnSchema(BaseModel):
    """工具返回值Schema"""
    type: ParameterType = Field(description="返回值类型")
    description: str = Field(description="返回值描述")
    example: Optional[Any] = Field(default=None, description="返回值示例")
    
    class Config:
        use_enum_values = True


class ToolExample(BaseModel):
    """工具使用示例"""
    description: str = Field(description="示例场景描述")
    input: Dict[str, Any] = Field(description="输入参数示例")
    output: Any = Field(description="预期输出示例")
    explanation: Optional[str] = Field(default=None, description="示例说明")


class ToolCategory(str, Enum):
    """工具分类"""
    DATA_PROCESSING = "data_processing"
    WEB_INTERACTION = "web_interaction"
    FILE_OPERATION = "file_operation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    UTILITY = "utility"


class ToolSchema(BaseModel):
    """
    工具Schema基类
    定义清晰、结构化、无歧义的工具描述规范
    """
    # 基本信息
    name: str = Field(description="工具名称，使用动词+名词格式，如：search_web, process_data")
    display_name: str = Field(description="工具显示名称，用户友好的名称")
    description: str = Field(description="工具功能的详细描述，说明工具的作用和适用场景")
    category: ToolCategory = Field(description="工具分类")
    version: str = Field(default="1.0.0", description="工具版本")
    
    # 参数定义
    parameters: List[ParameterSchema] = Field(
        default_factory=list,
        description="工具参数列表"
    )
    
    # 返回值定义
    returns: ReturnSchema = Field(description="工具返回值定义")
    
    # 使用示例
    examples: List[ToolExample] = Field(
        default_factory=list,
        description="工具使用示例列表"
    )
    
    # 使用场景说明
    use_cases: List[str] = Field(
        default_factory=list,
        description="工具的典型使用场景"
    )
    
    # 注意事项
    notes: Optional[List[str]] = Field(
        default=None,
        description="使用工具时的注意事项和限制"
    )
    
    # 依赖和前置条件
    prerequisites: Optional[List[str]] = Field(
        default=None,
        description="使用工具前需要满足的条件"
    )
    
    # 性能指标
    estimated_time: Optional[str] = Field(
        default=None,
        description="预计执行时间（如：1-2秒、5-10分钟）"
    )
    
    # 标签
    tags: List[str] = Field(
        default_factory=list,
        description="工具标签，用于搜索和分类"
    )
    
    class Config:
        use_enum_values = True
    
    def to_llm_format(self) -> Dict[str, Any]:
        """
        转换为LLM可理解的格式
        
        Returns:
            适合LLM使用的工具描述字典
        """
        # 构建参数描述
        params_desc = {}
        required_params = []
        
        for param in self.parameters:
            param_info = {
                "type": param.type,
                "description": param.description
            }
            
            if param.enum:
                param_info["enum"] = param.enum
            if param.default is not None:
                param_info["default"] = param.default
            if param.example is not None:
                param_info["example"] = param.example
            
            params_desc[param.name] = param_info
            
            if param.required:
                required_params.append(param.name)
        
        # 构建完整描述
        full_description = self.description
        
        if self.use_cases:
            full_description += f"\n\n适用场景：\n" + "\n".join(f"- {uc}" for uc in self.use_cases)
        
        if self.notes:
            full_description += f"\n\n注意事项：\n" + "\n".join(f"- {note}" for note in self.notes)
        
        return {
            "name": self.name,
            "description": full_description,
            "parameters": {
                "type": "object",
                "properties": params_desc,
                "required": required_params
            },
            "returns": {
                "type": self.returns.type,
                "description": self.returns.description
            }
        }
    
    def get_parameter_by_name(self, name: str) -> Optional[ParameterSchema]:
        """
        根据名称获取参数Schema
        
        Args:
            name: 参数名称
            
        Returns:
            参数Schema，如果不存在返回None
        """
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        验证输入参数
        
        Args:
            input_data: 输入参数字典
            
        Returns:
            (是否有效, 错误信息)
        """
        # 检查必填参数
        for param in self.parameters:
            if param.required and param.name not in input_data:
                return False, f"缺少必填参数: {param.name}"
        
        # 检查参数类型和约束
        for param_name, param_value in input_data.items():
            param_schema = self.get_parameter_by_name(param_name)
            if not param_schema:
                return False, f"未知参数: {param_name}"
            
            # 类型检查（简化版）
            if param_schema.type == ParameterType.STRING and not isinstance(param_value, str):
                return False, f"参数 {param_name} 应为字符串类型"
            elif param_schema.type == ParameterType.INTEGER and not isinstance(param_value, int):
                return False, f"参数 {param_name} 应为整数类型"
            elif param_schema.type == ParameterType.BOOLEAN and not isinstance(param_value, bool):
                return False, f"参数 {param_name} 应为布尔类型"
            elif param_schema.type == ParameterType.ARRAY and not isinstance(param_value, list):
                return False, f"参数 {param_name} 应为数组类型"
            
            # 枚举值检查
            if param_schema.enum and param_value not in param_schema.enum:
                return False, f"参数 {param_name} 的值必须是以下之一: {param_schema.enum}"
        
        return True, None
