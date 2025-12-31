#!/usr/bin/env python3
"""
Meta Agent 主入口
提供命令行交互界面
"""
import asyncio
import sys
import json
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 默认INFO级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 设置第三方库的日志级别为WARNING，避免显示过多信息
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

from meta_agent.core.skill_agent import SkillAgent


class AgentCLI:
    """命令行交互界面"""
    
    def __init__(self):
        self.agent = SkillAgent()
        self.running = True
        self.debug_mode = False  # 调试模式标志

    def print_banner(self):
        """打印欢迎信息"""
        print("=" * 60)
        print("  🤖 社交媒体文案生成助手")
        print("  基于 LangGraph + Agent Skills")
        print("=" * 60)
        print()

    def print_help(self):
        """打印帮助信息"""
        print("\n📚 可用命令:")
        print("  skills        - 查看可用技能")
        print("  tools         - 查看可用工具")
        print("  debug         - 切换调试模式")
        print("  reset         - 重置会话")
        print("  exit          - 退出程序")
        print("\n💡 直接输入任务描述即可开始对话")
        print()
        
    def print_skills(self):
        """打印可用技能"""
        skills = self.agent.get_available_skills()
        print(f"\n📦 可用技能 ({len(skills)} 个):")
        for skill in skills:
            print(f"  • {skill['name']}")
            print(f"    分类: {skill['category']}")
            print(f"    描述: {skill['description'][:80]}...")
            print()
            
    def print_tools(self):
        """打印可用工具"""
        tools = self.agent.get_available_tools()
        print(f"\n🔧 可用工具 ({len(tools)} 个):")
        for tool in tools:
            print(f"  • {tool['display_name']} ({tool['category']})")
            print(f"    {tool['description'][:80]}...")
            print()
            
    def print_session_info(self):
        """打印会话信息"""
        info = self.agent.get_session_info()
        print("\n📊 会话信息:")
        print(f"  会话ID: {info['session_id']}")
        print(f"  运行状态: {'运行中' if info['is_running'] else '空闲'}")
        print(f"  当前任务: {info.get('current_task', '无')}")
        print(f"  迭代次数: {info.get('iteration_count', 0)}")
        print(f"  已加载技能: {', '.join(info.get('loaded_skills', [])) or '无'}")
        print(f"  可用工具数: {info.get('available_tools_count', 0)}")
        
        if 'execution_stats' in info:
            stats = info['execution_stats']
            print(f"\n  执行统计:")
            print(f"    总执行次数: {stats.get('total_executions', 0)}")
            print(f"    成功次数: {stats.get('successful_executions', 0)}")
            print(f"    失败次数: {stats.get('failed_executions', 0)}")
        print()
        
    def print_history(self):
        """打印执行历史"""
        history = self.agent.get_execution_history()
        if not history:
            print("\n📜 执行历史: 暂无记录")
            return
            
        print(f"\n📜 执行历史 ({len(history)} 条):")
        for i, record in enumerate(history[-5:], 1):  # 只显示最近5条
            print(f"\n  {i}. {record.get('skill_name', 'Unknown')}")
            print(f"     状态: {'✅ 成功' if record.get('success') else '❌ 失败'}")
            print(f"     时间: {record.get('timestamp', 'Unknown')}")
            if record.get('error'):
                print(f"     错误: {record['error']}")
        print()
        
    async def print_capabilities(self):
        """打印 Agent 能力"""
        capabilities = await self.agent.explain_capabilities()
        print("\n🎯 Agent 能力说明:")
        print(f"  {capabilities['description']}")
        print("\n  核心特性:")
        for feature in capabilities['features']:
            print(f"    • {feature}")
        print(f"\n  可用技能数: {capabilities['available_skills']}")
        print(f"  可用工具数: {capabilities['available_tools']}")
        print(f"  技能分类: {', '.join(capabilities['skill_categories'])}")
        print(f"  工具分类: {', '.join(capabilities['tool_categories'])}")
        print()
        
    async def process_command(self, user_input: str):
        """处理用户命令"""
        command = user_input.strip().lower()
        
        if command == 'exit':
            print("\n👋 再见！")
            self.running = False
            return

        elif command == 'skills':
            self.print_skills()

        elif command == 'tools':
            self.print_tools()

        elif command == 'debug':
            self.toggle_debug_mode()

        elif command == 'reset':
            self.agent.reset_session()
            print("\n✅ 会话已重置")

        else:
            # 处理用户任务
            await self.process_task(user_input)

    def toggle_debug_mode(self):
        """切换调试模式"""
        self.debug_mode = not self.debug_mode

        if self.debug_mode:
            # 设置为DEBUG级别
            logging.getLogger().setLevel(logging.DEBUG)
            print("\n🐛 调试模式已开启 - 将显示详细的日志信息")
        else:
            # 设置为WARNING级别（不显示INFO和DEBUG）
            logging.getLogger().setLevel(logging.WARNING)
            print("\n✅ 调试模式已关闭 - 只显示警告和错误信息")
            
    async def process_task(self, user_input: str):
        """处理用户任务"""
        print("\n🤔 正在生成内容...")

        try:
            # 使用 Plan-Execute 模式（LangGraph 标准模式）
            result = await self.agent.process_with_plan_execute(
                user_prompt=user_input,
                save_to_local=False  # 默认不保存到本地，可根据需要修改
            )

            # 打印最终输出
            if result.get("final_output"):
                print("\n" + "=" * 60)
                print("🤖:")
                print(result["final_output"])
                print("=" * 60)

            # # 打印执行计划
            # if result.get("plan"):
            #     print("\n📋 执行计划:")
            #     for step in result["plan"]:
            #         status_icon = "✅" if step["status"] == "completed" else "⏳" if step["status"] == "in_progress" else "❌" if step["status"] == "failed" else "⏸️"
            #         print(f"  {status_icon} {step['step_id']}. {step['worker_name']}")
            #         if step.get("result"):
            #             print(f"     状态: {step['status']}")

            # 打印步骤结果
            if result.get("step_results"):
                for step_id, step_result in result["step_results"].items():
                    if step_result.get("success"):
                        content = step_result.get("content", "")
                        print(content)  # 显示完整内容，不再截断
                print("-" * 60)

            # 打印质量检查报告（放在最后）
            if result.get("qa_report"):
                print("\n" + "=" * 60)
                print("📊 质量检查报告")
                print("=" * 60)
                print(result["qa_report"])
                print("=" * 60)

            # 打印保存信息
            if result.get("output_directory"):
                print(f"\n💾 内容已保存到: {result['output_directory']}")

            # 打印错误信息
            if result.get("errors"):
                print("\n❌ 错误信息:")
                for error in result["errors"]:
                    print(f"  - {error}")

            # 如果需要查看完整结果，可以取消注释下面的代码
            # print("\n" + "=" * 60)
            # print("📊 完整执行结果:")
            # print("=" * 60)
            # print(json.dumps(result, indent=2, ensure_ascii=False))
            # print("=" * 60)

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            
        print()
        
    async def run(self):
        """运行命令行界面"""
        self.print_banner()
        self.print_help()
        
        while self.running:
            try:
                user_input = input("💬 您: ").strip()
                
                if not user_input:
                    continue
                    
                await self.process_command(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except EOFError:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                

async def main():
    """主函数"""
    cli = AgentCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
