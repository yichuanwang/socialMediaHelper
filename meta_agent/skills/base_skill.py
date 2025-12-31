"""
BaseSkill - 技能基类
定义技能的标准接口和数据结构
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from pathlib import Path


class SkillMetadata(BaseModel):
    """技能元数据（第一层信息）"""
    name: str = Field(description="技能名称")
    description: str = Field(description="技能简短描述")
    category: str = Field(description="技能分类")
    version: str = Field(default="1.0.0", description="技能版本")
    author: Optional[str] = Field(default=None, description="作者")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    required_tools: List[str] = Field(default_factory=list, description="所需工具列表")


class SkillResource(BaseModel):
    """技能资源"""
    name: str = Field(description="资源名称")
    path: str = Field(description="资源路径")
    type: str = Field(description="资源类型：script/template/reference")
    description: Optional[str] = Field(default=None, description="资源描述")


class Skill(BaseModel):
    """
    技能完整定义
    实现Claude Skills的渐进式披露机制
    """
    # 第一层：元数据（始终加载）
    metadata: SkillMetadata
    
    # 第二层：详细说明（按需加载）
    detailed_description: Optional[str] = Field(
        default=None,
        description="详细功能说明"
    )
    
    use_cases: Optional[List[str]] = Field(
        default=None,
        description="使用场景列表"
    )
    
    prerequisites: Optional[List[str]] = Field(
        default=None,
        description="前置条件"
    )
    
    # 第三层：资源文件（运行时加载）
    resources: Optional[List[SkillResource]] = Field(
        default=None,
        description="关联的资源文件"
    )
    
    # 技能所在目录
    skill_dir: Optional[Path] = Field(
        default=None,
        description="技能目录路径"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_summary(self) -> str:
        """
        获取技能摘要（用于上下文）
        
        Returns:
            技能摘要字符串
        """
        return f"{self.metadata.name}: {self.metadata.description}"
    
    def get_full_description(self) -> str:
        """
        获取完整描述
        
        Returns:
            完整描述字符串
        """
        parts = [
            f"# {self.metadata.name}",
            f"\n**Category**: {self.metadata.category}",
            f"\n**Version**: {self.metadata.version}",
            f"\n\n## Description\n{self.metadata.description}"
        ]
        
        if self.detailed_description:
            parts.append(f"\n\n## Detailed Description\n{self.detailed_description}")
            
        if self.use_cases:
            parts.append("\n\n## Use Cases")
            for case in self.use_cases:
                parts.append(f"\n- {case}")
                
        if self.prerequisites:
            parts.append("\n\n## Prerequisites")
            for prereq in self.prerequisites:
                parts.append(f"\n- {prereq}")
                
        return "".join(parts)
    
    def load_resource(self, resource_name: str) -> Optional[str]:
        """
        加载资源文件内容
        
        Args:
            resource_name: 资源名称
            
        Returns:
            资源内容或None
        """
        if not self.resources or not self.skill_dir:
            return None
            
        for resource in self.resources:
            if resource.name == resource_name:
                resource_path = self.skill_dir / resource.path
                if resource_path.exists():
                    return resource_path.read_text(encoding='utf-8')
                    
        return None
    
    def list_resources(self) -> List[str]:
        """
        列出所有资源名称
        
        Returns:
            资源名称列表
        """
        if not self.resources:
            return []
        return [r.name for r in self.resources]


class SkillContext(BaseModel):
    """技能执行上下文"""
    skill_name: str
    task: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    loaded_resources: Dict[str, str] = Field(default_factory=dict)
