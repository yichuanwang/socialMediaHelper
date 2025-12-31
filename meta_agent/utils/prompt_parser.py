"""
Prompt解析器 - 分析用户输入的prompt，提取任务信息
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
import logging


logger = logging.getLogger(__name__)


@dataclass
class ParsedPrompt:
    """解析后的Prompt结构"""
    # 原始prompt
    original_prompt: str
    
    # 任务目标
    task_goal: str
    
    # 任务类型
    task_type: str  # analysis/generation/search/processing/mixed
    
    # 所需能力/技能
    required_skills: List[str]
    
    # 关键实体
    entities: Dict[str, List[str]]  # 类型 -> 实体列表
    
    # 约束条件
    constraints: List[str]
    
    # 优先级
    priority: str  # high/medium/low
    
    # 预估复杂度
    complexity: str  # simple/medium/complex
    
    # 额外元数据
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（用于JSON序列化）

        Returns:
            可序列化的字典
        """
        return {
            "original_prompt": self.original_prompt,
            "task_goal": self.task_goal,
            "task_type": self.task_type,
            "required_skills": self.required_skills,
            "entities": self.entities,
            "constraints": self.constraints,
            "priority": self.priority,
            "complexity": self.complexity,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParsedPrompt':
        """
        从字典创建ParsedPrompt对象

        Args:
            data: 字典数据

        Returns:
            ParsedPrompt对象
        """
        return cls(
            original_prompt=data.get("original_prompt", ""),
            task_goal=data.get("task_goal", ""),
            task_type=data.get("task_type", "mixed"),
            required_skills=data.get("required_skills", []),
            entities=data.get("entities", {}),
            constraints=data.get("constraints", []),
            priority=data.get("priority", "medium"),
            complexity=data.get("complexity", "medium"),
            metadata=data.get("metadata", {})
        )


class PromptParser:
    """
    Prompt解析器
    分析用户输入，提取任务信息和所需技能
    """
    
    def __init__(self):
        """初始化解析器"""
        # 任务类型关键词映射
        self.task_type_keywords = {
            "analysis": ["分析", "analyze", "研究", "study", "调查", "investigate"],
            "generation": ["生成", "generate", "创建", "create", "编写", "write", "构建", "build"],
            "search": ["搜索", "search", "查找", "find", "检索", "retrieve"],
            "processing": ["处理", "process", "转换", "convert", "格式化", "format"],
            "explanation": ["解释", "explain", "说明", "describe", "介绍", "introduce"]
        }
        
        # 技能关键词映射
        self.skill_keywords = {
            "code_analysis": ["代码分析", "code analysis", "代码审查", "code review"],
            "web_search": ["网络搜索", "web search", "搜索网页", "search web"],
            "document_generation": ["文档生成", "document generation", "编写文档", "write document", "技术文档", "technical document", "文档", "documentation"],
            "text_processing": ["文本处理", "text processing", "文本分析", "text analysis"],
            "file_operation": ["文件操作", "file operation", "读写文件", "read write file"]
        }
        
        # 约束关键词
        self.constraint_keywords = [
            "必须", "must", "不能", "cannot", "should", "应该",
            "限制", "limit", "要求", "require", "禁止", "forbid"
        ]
        
        logger.debug("PromptParser initialized")
    
    def parse(self, prompt: str) -> ParsedPrompt:
        """
        解析prompt
        
        Args:
            prompt: 用户输入的prompt
            
        Returns:
            解析后的Prompt结构
        """
        try:
            # 提取任务目标
            task_goal = self._extract_task_goal(prompt)
            
            # 识别任务类型
            task_type = self._identify_task_type(prompt)
            
            # 提取所需技能
            required_skills = self._extract_required_skills(prompt)
            
            # 提取实体
            entities = self._extract_entities(prompt)
            
            # 提取约束条件
            constraints = self._extract_constraints(prompt)
            
            # 评估优先级
            priority = self._assess_priority(prompt)
            
            # 评估复杂度
            complexity = self._assess_complexity(prompt, required_skills)
            
            # 构建元数据
            metadata = {
                "prompt_length": len(prompt),
                "word_count": len(prompt.split()),
                "has_code_block": "```" in prompt,
                "has_url": bool(re.search(r'https?://', prompt))
            }
            
            parsed = ParsedPrompt(
                original_prompt=prompt,
                task_goal=task_goal,
                task_type=task_type,
                required_skills=required_skills,
                entities=entities,
                constraints=constraints,
                priority=priority,
                complexity=complexity,
                metadata=metadata
            )
            
            logger.debug(f"Prompt parsed successfully: type={task_type}, complexity={complexity}")
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse prompt: {e}")
            # 返回基本解析结果
            return ParsedPrompt(
                original_prompt=prompt,
                task_goal=prompt[:100] if len(prompt) > 100 else prompt,
                task_type="mixed",
                required_skills=[],
                entities={},
                constraints=[],
                priority="medium",
                complexity="medium",
                metadata={}
            )
    
    def _extract_task_goal(self, prompt: str) -> str:
        """
        提取任务目标
        
        Args:
            prompt: 原始prompt
            
        Returns:
            任务目标描述
        """
        # 简化版：取第一句话或前100个字符
        sentences = re.split(r'[。！？\n]', prompt)
        first_sentence = sentences[0].strip() if sentences else prompt
        
        if len(first_sentence) > 100:
            return first_sentence[:100] + "..."
        return first_sentence
    
    def _identify_task_type(self, prompt: str) -> str:
        """
        识别任务类型
        
        Args:
            prompt: 原始prompt
            
        Returns:
            任务类型
        """
        prompt_lower = prompt.lower()
        type_scores = {}
        
        for task_type, keywords in self.task_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                type_scores[task_type] = score
        
        if not type_scores:
            return "mixed"
        
        # 返回得分最高的类型
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def _extract_required_skills(self, prompt: str) -> List[str]:
        """
        提取所需技能
        
        Args:
            prompt: 原始prompt
            
        Returns:
            技能列表
        """
        prompt_lower = prompt.lower()
        skills = []
        
        for skill, keywords in self.skill_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                skills.append(skill)
        
        # 如果没有匹配到具体技能，根据任务类型推断
        if not skills:
            task_type = self._identify_task_type(prompt)
            if task_type == "analysis":
                skills.append("code_analysis")
            elif task_type == "generation":
                skills.append("document_generation")
            elif task_type == "search":
                skills.append("web_search")
        
        return skills
    
    def _extract_entities(self, prompt: str) -> Dict[str, List[str]]:
        """
        提取关键实体
        
        Args:
            prompt: 原始prompt
            
        Returns:
            实体字典
        """
        entities = {
            "files": [],
            "urls": [],
            "numbers": [],
            "dates": []
        }
        
        # 提取文件路径
        file_patterns = [
            r'[\w/\\]+\.\w+',  # 文件路径
            r'`([^`]+\.\w+)`'  # 反引号中的文件
        ]
        for pattern in file_patterns:
            matches = re.findall(pattern, prompt)
            entities["files"].extend(matches)
        
        # 提取URL
        url_pattern = r'https?://[^\s]+'
        entities["urls"] = re.findall(url_pattern, prompt)
        
        # 提取数字
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        entities["numbers"] = re.findall(number_pattern, prompt)
        
        # 提取日期（简化版）
        date_pattern = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        entities["dates"] = re.findall(date_pattern, prompt)
        
        # 移除空列表
        return {k: v for k, v in entities.items() if v}
    
    def _extract_constraints(self, prompt: str) -> List[str]:
        """
        提取约束条件
        
        Args:
            prompt: 原始prompt
            
        Returns:
            约束列表
        """
        constraints = []
        sentences = re.split(r'[。！？\n]', prompt)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence for keyword in self.constraint_keywords):
                if sentence and len(sentence) > 5:
                    constraints.append(sentence)
        
        return constraints
    
    def _assess_priority(self, prompt: str) -> str:
        """
        评估优先级
        
        Args:
            prompt: 原始prompt
            
        Returns:
            优先级 (high/medium/low)
        """
        prompt_lower = prompt.lower()
        
        # 高优先级关键词
        high_priority_keywords = ["紧急", "urgent", "立即", "immediately", "asap", "重要", "critical"]
        if any(keyword in prompt_lower for keyword in high_priority_keywords):
            return "high"
        
        # 低优先级关键词
        low_priority_keywords = ["有空", "when free", "不急", "no rush", "可以慢慢", "later"]
        if any(keyword in prompt_lower for keyword in low_priority_keywords):
            return "low"
        
        return "medium"
    
    def _assess_complexity(self, prompt: str, required_skills: List[str]) -> str:
        """
        评估复杂度
        
        Args:
            prompt: 原始prompt
            required_skills: 所需技能列表
            
        Returns:
            复杂度 (simple/medium/complex)
        """
        # 基于多个因素评估
        score = 0
        
        # 因素1：prompt长度
        if len(prompt) > 500:
            score += 2
        elif len(prompt) > 200:
            score += 1
        
        # 因素2：所需技能数量
        score += len(required_skills)
        
        # 因素3：约束条件数量
        constraints = self._extract_constraints(prompt)
        score += len(constraints) // 2
        
        # 因素4：是否包含代码
        if "```" in prompt:
            score += 1
        
        # 因素5：多步骤任务
        step_keywords = ["步骤", "step", "然后", "then", "接着", "next", "最后", "finally"]
        if sum(1 for keyword in step_keywords if keyword in prompt.lower()) >= 2:
            score += 2
        
        # 根据得分判断复杂度
        if score >= 5:
            return "complex"
        elif score >= 2:
            return "medium"
        else:
            return "simple"
    
    def suggest_skills(self, parsed_prompt: ParsedPrompt) -> List[str]:
        """
        根据解析结果推荐技能
        
        Args:
            parsed_prompt: 解析后的prompt
            
        Returns:
            推荐的技能列表
        """
        suggestions = set(parsed_prompt.required_skills)
        
        # 根据任务类型添加建议
        if parsed_prompt.task_type == "analysis":
            suggestions.add("code_analysis")
        elif parsed_prompt.task_type == "generation":
            suggestions.add("document_generation")
        elif parsed_prompt.task_type == "search":
            suggestions.add("web_search")
        
        # 根据实体添加建议
        if "files" in parsed_prompt.entities:
            suggestions.add("file_operation")
        if "urls" in parsed_prompt.entities:
            suggestions.add("web_search")
        
        return list(suggestions)
    
    def format_summary(self, parsed_prompt: ParsedPrompt) -> str:
        """
        格式化解析摘要
        
        Args:
            parsed_prompt: 解析后的prompt
            
        Returns:
            格式化的摘要字符串
        """
        lines = [
            "=== Prompt解析摘要 ===",
            f"任务目标: {parsed_prompt.task_goal}",
            f"任务类型: {parsed_prompt.task_type}",
            f"复杂度: {parsed_prompt.complexity}",
            f"优先级: {parsed_prompt.priority}",
        ]
        
        if parsed_prompt.required_skills:
            lines.append(f"所需技能: {', '.join(parsed_prompt.required_skills)}")
        
        if parsed_prompt.constraints:
            lines.append(f"约束条件: {len(parsed_prompt.constraints)}个")
        
        if parsed_prompt.entities:
            entity_summary = ", ".join(f"{k}:{len(v)}" for k, v in parsed_prompt.entities.items())
            lines.append(f"关键实体: {entity_summary}")
        
        return "\n".join(lines)
