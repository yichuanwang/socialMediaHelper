"""
SkillLoader - 技能加载器
实现技能的渐进式加载机制
"""
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import re

from meta_agent.skills.base_skill import (
    Skill,
    SkillMetadata,
    SkillResource
)


class SkillLoader:
    """
    技能加载器
    实现Claude Skills的渐进式披露机制
    """
    
    def __init__(self, skills_dir: Path):
        """
        初始化技能加载器
        
        Args:
            skills_dir: 技能目录路径
        """
        self.skills_dir = Path(skills_dir)
        
    def load_skill_metadata(self, skill_name: str) -> Optional[SkillMetadata]:
        """
        加载技能元数据（第一层）
        只加载YAML frontmatter部分
        
        Args:
            skill_name: 技能名称
            
        Returns:
            技能元数据或None
        """
        skill_file = self._find_skill_file(skill_name)
        if not skill_file:
            return None
            
        try:
            content = skill_file.read_text(encoding='utf-8')
            metadata_dict = self._parse_frontmatter(content)
            
            if not metadata_dict:
                return None
                
            return SkillMetadata(**metadata_dict)
        except Exception as e:
            print(f"Error loading skill metadata for {skill_name}: {e}")
            return None
    
    def load_skill_details(self, skill_name: str) -> Optional[Skill]:
        """
        加载技能详细信息（第二层）
        加载完整的SKILL.md内容
        
        Args:
            skill_name: 技能名称
            
        Returns:
            完整的Skill对象或None
        """
        skill_file = self._find_skill_file(skill_name)
        if not skill_file:
            return None
            
        try:
            content = skill_file.read_text(encoding='utf-8')
            
            # 解析frontmatter
            metadata_dict = self._parse_frontmatter(content)
            if not metadata_dict:
                return None
                
            metadata = SkillMetadata(**metadata_dict)
            
            # 解析markdown内容
            body = self._extract_body(content)
            sections = self._parse_markdown_sections(body)
            
            # 构建Skill对象
            skill = Skill(
                metadata=metadata,
                detailed_description=sections.get('description'),
                use_cases=sections.get('use_cases', []),
                prerequisites=sections.get('prerequisites', []),
                resources=self._parse_resources(sections.get('resources', [])),
                skill_dir=skill_file.parent
            )
            
            return skill
        except Exception as e:
            print(f"Error loading skill details for {skill_name}: {e}")
            return None
    
    def load_skill_summary(self, skill_name: str) -> Optional[str]:
        """
        加载技能的Summary部分
        用于Orchestrator-Worker架构中的Orchestrator层

        Args:
            skill_name: 技能名称

        Returns:
            Summary内容或None
        """
        skill_file = self._find_skill_file(skill_name)
        if not skill_file:
            return None

        try:
            content = skill_file.read_text(encoding='utf-8')

            # 去除frontmatter
            body = self._extract_body(content)

            # 提取Summary章节
            # 匹配 ## Summary 到下一个 ## 或文件结尾
            summary_match = re.search(
                r'##\s*Summary\s*\n(.*?)(?=\n##|\Z)',
                body,
                re.DOTALL | re.IGNORECASE
            )

            if summary_match:
                return summary_match.group(1).strip()

            return None

        except Exception as e:
            print(f"Error loading skill summary for {skill_name}: {e}")
            return None

    def load_skill_with_metadata(self, skill_name: str) -> Optional[Dict[str, str]]:
        """
        加载技能的Frontmatter和完整内容
        用于在system prompt中同时展示元数据和详细说明

        Args:
            skill_name: 技能名称

        Returns:
            包含metadata和summary的字典，或None
        """
        skill_file = self._find_skill_file(skill_name)
        if not skill_file:
            return None

        try:
            content = skill_file.read_text(encoding='utf-8')

            # 解析frontmatter
            metadata_dict = self._parse_frontmatter(content)
            if not metadata_dict:
                return None

            # 直接使用去除frontmatter后的完整body内容
            # 不需要正则匹配，更简单、更高效、更鲁棒
            body = self._extract_body(content)

            return {
                "metadata": metadata_dict,
                "summary": body  # 直接返回完整的body内容
            }

        except Exception as e:
            print(f"Error loading skill with metadata for {skill_name}: {e}")
            return None

    def load_skill_resource(
        self,
        skill_name: str,
        resource_name: str
    ) -> Optional[str]:
        """
        加载技能资源（第三层）
        按需加载具体的资源文件
        
        Args:
            skill_name: 技能名称
            resource_name: 资源名称
            
        Returns:
            资源内容或None
        """
        skill = self.load_skill_details(skill_name)
        if not skill:
            return None
            
        return skill.load_resource(resource_name)
    
    def _find_skill_file(self, skill_name: str) -> Optional[Path]:
        """
        查找技能文件
        
        Args:
            skill_name: 技能名称
            
        Returns:
            技能文件路径或None
        """
        # 尝试多种可能的路径
        possible_paths = [
            self.skills_dir / skill_name / "SKILL.md",
            self.skills_dir / f"{skill_name}.md",
            self.skills_dir / "examples" / skill_name / "SKILL.md",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        return None
    
    def _parse_frontmatter(self, content: str) -> Optional[Dict[str, Any]]:
        """
        解析YAML frontmatter
        
        Args:
            content: 文件内容
            
        Returns:
            元数据字典或None
        """
        # 匹配 --- 包围的YAML内容
        pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(pattern, content, re.DOTALL)
        
        if not match:
            return None
            
        yaml_content = match.group(1)
        try:
            return yaml.safe_load(yaml_content)
        except yaml.YAMLError:
            return None
    
    def _extract_body(self, content: str) -> str:
        """
        提取markdown正文（去除frontmatter）
        
        Args:
            content: 文件内容
            
        Returns:
            正文内容
        """
        pattern = r'^---\s*\n.*?\n---\s*\n'
        return re.sub(pattern, '', content, count=1, flags=re.DOTALL)
    
    def _parse_markdown_sections(self, body: str) -> Dict[str, Any]:
        """
        解析markdown章节
        
        Args:
            body: markdown正文
            
        Returns:
            章节字典
        """
        sections = {}
        
        # 提取描述部分（第一个##之前的内容）
        desc_match = re.search(r'^(.*?)(?=\n##|\Z)', body, re.DOTALL)
        if desc_match:
            sections['description'] = desc_match.group(1).strip()
        
        # 提取Use Cases
        use_cases_match = re.search(
            r'##\s*Use Cases\s*\n(.*?)(?=\n##|\Z)',
            body,
            re.DOTALL | re.IGNORECASE
        )
        if use_cases_match:
            use_cases_text = use_cases_match.group(1)
            sections['use_cases'] = self._parse_list_items(use_cases_text)
        
        # 提取Prerequisites
        prereq_match = re.search(
            r'##\s*Prerequisites\s*\n(.*?)(?=\n##|\Z)',
            body,
            re.DOTALL | re.IGNORECASE
        )
        if prereq_match:
            prereq_text = prereq_match.group(1)
            sections['prerequisites'] = self._parse_list_items(prereq_text)
        
        # 提取Resources
        resources_match = re.search(
            r'##\s*Resources\s*\n(.*?)(?=\n##|\Z)',
            body,
            re.DOTALL | re.IGNORECASE
        )
        if resources_match:
            resources_text = resources_match.group(1)
            sections['resources'] = self._parse_list_items(resources_text)
        
        return sections
    
    def _parse_list_items(self, text: str) -> list:
        """
        解析列表项
        
        Args:
            text: 包含列表的文本
            
        Returns:
            列表项列表
        """
        items = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                item = line[1:].strip()
                if item:
                    items.append(item)
        return items
    
    def _parse_resources(self, resource_lines: list) -> list:
        """
        解析资源列表
        
        Args:
            resource_lines: 资源行列表
            
        Returns:
            SkillResource对象列表
        """
        resources = []
        for line in resource_lines:
            # 解析markdown链接格式: [name](path)
            match = re.match(r'\[([^\]]+)\]\(([^\)]+)\)', line)
            if match:
                name = match.group(1)
                path = match.group(2)
                
                # 推断资源类型
                resource_type = "reference"
                if path.endswith('.py'):
                    resource_type = "script"
                elif path.endswith('.j2') or path.endswith('.template'):
                    resource_type = "template"
                
                resources.append(SkillResource(
                    name=name,
                    path=path,
                    type=resource_type
                ))
        
        return resources
    
    def list_available_skills(self) -> list:
        """
        列出所有可用的技能
        
        Returns:
            技能名称列表
        """
        skills = []
        
        if not self.skills_dir.exists():
            return skills
        
        # 扫描技能目录
        for item in self.skills_dir.iterdir():
            if item.is_dir():
                skill_file = item / "SKILL.md"
                if skill_file.exists():
                    skills.append(item.name)
        
        # 扫描examples目录
        examples_dir = self.skills_dir / "examples"
        if examples_dir.exists():
            for item in examples_dir.iterdir():
                if item.is_dir():
                    skill_file = item / "SKILL.md"
                    if skill_file.exists():
                        skills.append(item.name)
        
        return skills
