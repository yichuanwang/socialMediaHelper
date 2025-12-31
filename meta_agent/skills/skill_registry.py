"""
SkillRegistry - 技能注册表
管理所有可用技能的元数据和加载
"""
from typing import Dict, List, Optional
from pathlib import Path

from meta_agent.skills.base_skill import Skill, SkillMetadata
from meta_agent.skills.skill_loader import SkillLoader


class SkillRegistry:
    """
    技能注册表
    实现技能的注册、查询和匹配
    """
    
    def __init__(self, skills_dir: Optional[Path] = None):
        """
        初始化技能注册表
        
        Args:
            skills_dir: 技能目录路径，默认为当前目录下的skills
        """
        if skills_dir is None:
            # 默认使用 meta_agent/skills 目录
            skills_dir = Path(__file__).parent
        
        self.skills_dir = Path(skills_dir)
        self.loader = SkillLoader(self.skills_dir)
        
        # 技能元数据缓存（第一层，始终加载）
        self._metadata_cache: Dict[str, SkillMetadata] = {}
        
        # 技能详情缓存（第二层，按需加载）
        self._skill_cache: Dict[str, Skill] = {}
        
        # 初始化时加载所有技能元数据
        self._load_all_metadata()
    
    def _load_all_metadata(self):
        """加载所有技能的元数据"""
        available_skills = self.loader.list_available_skills()
        
        for skill_name in available_skills:
            metadata = self.loader.load_skill_metadata(skill_name)
            if metadata:
                self._metadata_cache[skill_name] = metadata
    
    def get_skill_metadata(self, skill_name: str) -> Optional[SkillMetadata]:
        """
        获取技能元数据
        
        Args:
            skill_name: 技能名称
            
        Returns:
            技能元数据或None
        """
        return self._metadata_cache.get(skill_name)
    
    def get_skill(self, skill_name: str) -> Optional[Skill]:
        """
        获取完整的技能对象
        使用缓存机制
        
        Args:
            skill_name: 技能名称
            
        Returns:
            Skill对象或None
        """
        # 先检查缓存
        if skill_name in self._skill_cache:
            return self._skill_cache[skill_name]
        
        # 加载技能详情
        skill = self.loader.load_skill_details(skill_name)
        if skill:
            self._skill_cache[skill_name] = skill
        
        return skill
    
    def list_all_skills(self) -> List[str]:
        """
        列出所有可用技能名称
        
        Returns:
            技能名称列表
        """
        return list(self._metadata_cache.keys())
    
    
    def get_all_skills(self) -> List[SkillMetadata]:
        """
        获取所有技能的元数据列表
        
        Returns:
            技能元数据列表
        """
        return list(self._metadata_cache.values())
    
    def search_skills(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[SkillMetadata]:
        """
        搜索技能
        
        Args:
            query: 搜索关键词
            category: 技能分类过滤
            tags: 标签过滤
            
        Returns:
            匹配的技能元数据列表
        """
        results = []
        query_lower = query.lower()
        
        for skill_name, metadata in self._metadata_cache.items():
            # 分类过滤
            if category and metadata.category != category:
                continue
            
            # 标签过滤
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            # 关键词匹配
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append(metadata)
        
        return results
    
    def match_skills_for_task(self, task_description: str) -> List[str]:
        """
        根据任务描述匹配相关技能
        简单实现：基于关键词匹配
        
        Args:
            task_description: 任务描述
            
        Returns:
            匹配的技能名称列表
        """
        matched_skills = []
        task_lower = task_description.lower()
        
        # 定义关键词到技能的映射
        keyword_mappings = {
            'data': ['data_analysis'],
            'analyze': ['data_analysis'],
            'search': ['web_search'],
            'web': ['web_search'],
            'code': ['code_generation'],
            'generate': ['code_generation'],
            'program': ['code_generation'],
        }
        
        # 基于关键词匹配
        for keyword, skills in keyword_mappings.items():
            if keyword in task_lower:
                for skill in skills:
                    if skill in self._metadata_cache and skill not in matched_skills:
                        matched_skills.append(skill)
        
        # 如果没有匹配到，返回所有技能
        if not matched_skills:
            matched_skills = self.list_all_skills()
        
        return matched_skills
    
    def get_skill_summary(self, skill_name: str) -> Optional[str]:
        """
        获取技能摘要（用于上下文）
        
        Args:
            skill_name: 技能名称
            
        Returns:
            技能摘要或None
        """
        metadata = self.get_skill_metadata(skill_name)
        if metadata:
            return f"{metadata.name}: {metadata.description}"
        return None
    
    def get_all_skill_summaries(self) -> Dict[str, str]:
        """
        获取所有技能的摘要
        用于加载到全局状态
        
        Returns:
            技能名称到摘要的映射
        """
        summaries = {}
        for skill_name, metadata in self._metadata_cache.items():
            summaries[skill_name] = f"{metadata.name}: {metadata.description}"
        return summaries
    
    def get_skills_by_category(self, category: str) -> List[str]:
        """
        按分类获取技能
        
        Args:
            category: 分类名称
            
        Returns:
            技能名称列表
        """
        return [
            skill_name for skill_name, metadata in self._metadata_cache.items()
            if metadata.category == category
        ]
    
    def reload_skills(self):
        """重新加载所有技能"""
        self._metadata_cache.clear()
        self._skill_cache.clear()
        self._load_all_metadata()
    
    def register_skill(self, skill: Skill):
        """
        手动注册技能
        
        Args:
            skill: Skill对象
        """
        skill_name = skill.metadata.name
        self._metadata_cache[skill_name] = skill.metadata
        self._skill_cache[skill_name] = skill
    
    def unregister_skill(self, skill_name: str):
        """
        注销技能
        
        Args:
            skill_name: 技能名称
        """
        self._metadata_cache.pop(skill_name, None)
        self._skill_cache.pop(skill_name, None)
    
    def get_skill_count(self) -> int:
        """
        获取技能数量
        
        Returns:
            技能数量
        """
        return len(self._metadata_cache)
    
    def validate_skill(self, skill_name: str) -> bool:
        """
        验证技能是否有效
        
        Args:
            skill_name: 技能名称
            
        Returns:
            是否有效
        """
        metadata = self.get_skill_metadata(skill_name)
        if not metadata:
            return False
        
        # 检查必要字段
        if not metadata.name or not metadata.description:
            return False
        
        return True

    def has_skill(self, skill_name: str) -> bool:
        """
        检查技能是否存在
        
        Args:
            skill_name: 技能名称
            
        Returns:
            技能是否存在
        """
        return skill_name in self._metadata_cache
