import gradio as gr
import torch
from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer
import os
from utils.helpers import Logger

class SimpleMedicalUI:
    """
    简单的医学影像对话UI界面
    """
    
    def __init__(self, model_path):
        """
        初始化UI界面
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        self.load_model()
        
        Logger.info("医学影像对话UI初始化完成")
    
    def load_model(self):
        """加载模型"""
        Logger.info("正在加载模型...")

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["UNSLOTH_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["UNSLOTH_DISABLE_TELEMETRY"] = "1"

        # 禁用网络连接
        os.environ["NO_PROXY"] = "*"
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        
        print("🔌 强制离线模式已启用")
        
        try:
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=self.model_path,
                load_in_4bit=True,
                local_files_only=True
            )
            FastVisionModel.for_inference(self.model)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def generate_response(self, image, question, max_new_tokens=256, temperature=1.5, min_p=0.2):
        """
        生成模型响应 - 使用您测试程序中的方法
        
        Args:
            image: 上传的图像
            question: 用户问题
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            min_p: 最小概率阈值
            
        Returns:
            response: 模型响应
        """
        if image is None:
            return "请先上传一张医学影像图片。"
        
        try:
            # 直接使用Gradio返回的图像，它已经是PIL格式
            # Gradio的Image组件(type="pil")会直接返回PIL.Image对象
            pil_image = image
            
            # 确保图像是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 使用您测试程序中的指令
            if not question or question.strip() == "":
                instruction = "你是一名专业的放射科医生。请准确描述你在图片中看到的内容。"
            else:
                instruction = question
            
            # 使用您测试程序中的消息格式
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction}
                    ]
                }
            ]
            
            # 使用您测试程序中的方法
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            # 直接传递PIL图像给tokenizer
            inputs = self.tokenizer(
                pil_image,
                input_text,
                add_special_tokens=False,  # 如您测试程序中所述
                return_tensors='pt',
            ).to(self.device)
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=temperature,
                    min_p=min_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的响应部分（去掉输入）
            if input_text in response:
                response = response.replace(input_text, "").strip()
            
            # 清理响应文本
            response = self.clean_response(response)
            
            return response
            
        except Exception as e:
            error_msg = f"生成响应时出错: {str(e)}"
            Logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    def clean_response(self, response):
        """
        清理模型响应，移除不必要的标记和重复内容
        
        Args:
            response: 原始模型响应
            
        Returns:
            cleaned_response: 清理后的响应
        """
        # 移除特殊标记
        special_tokens = ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        for token in special_tokens:
            response = response.replace(token, "")
        
        return response.strip()
    
    def test_model(self):
        """
        测试模型是否能正常工作
        """
        try:
            # 创建一个简单的测试图像
            test_image = Image.new('RGB', (256, 256), color=(100, 100, 100))
            
            # 测试问题
            test_question = "描述这张图片"
            
            # 生成响应
            response = self.generate_response(test_image, test_question)
            
            if response and "出错" not in response:
                print("✅ 模型测试成功!")
                print(f"测试响应: {response}")
                return True
            else:
                print("❌ 模型测试失败!")
                return False
                
        except Exception as e:
            print(f"❌ 模型测试出错: {e}")
            return False
    
    def create_interface(self):
        """创建Gradio界面"""
        
        # 定义CSS样式
        css = """
        .gradio-container {
            max-width: 1000px !important;
        }
        .medical-title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
            font-family: Arial, sans-serif;
        }
        """
        
        # 示例问题列表
        example_questions = [
            "你是一名专业的放射科医生。请准确描述你在图片中看到的内容。",
            "请描述这张医学影像中的异常发现。",
            "根据这张影像，你的诊断意见是什么？",
            "请详细描述影像中的解剖结构和可能的病理变化。",
            "这张影像是否显示任何异常？如果有，请描述。",
            "影像中的这些特征可能表示什么疾病？"
        ]
        
        with gr.Blocks(css=css, title="医学影像AI助手") as interface:
            gr.Markdown(
                """
                # 🏥 医学影像AI助手
                **上传医学影像图片，与专业的AI放射科医生进行对话**
                """,
                elem_classes="medical-title"
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 图像上传区域 - 使用pil类型，直接返回PIL图像
                    image_input = gr.Image(
                        label="上传医学影像",
                        type="pil",  # 直接返回PIL图像对象
                        height=300
                    )
                    
                    # 示例问题
                    gr.Markdown("### 💡 示例问题")
                    
                    # 存储示例按钮的列表
                    example_buttons = []
                    
                    # 创建示例问题按钮但不立即绑定事件
                    for question in example_questions:
                        btn = gr.Button(
                            question, 
                            size="sm"
                        )
                        example_buttons.append((btn, question))
                    
                    # 参数调整
                    with gr.Accordion("⚙️ 高级设置", open=False):
                        max_tokens = gr.Slider(
                            minimum=64,
                            maximum=512,
                            value=256,
                            step=32,
                            label="最大生成长度"
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="生成温度"
                        )
                        min_p = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=0.2,
                            step=0.05,
                            label="最小概率阈值"
                        )
                
                with gr.Column(scale=2):
                    # 问题输入
                    question_input = gr.Textbox(
                        label="输入您的问题",
                        placeholder="请输入关于这张医学影像的问题...",
                        lines=3
                    )
                    
                    # 提交按钮
                    submit_btn = gr.Button("分析影像", variant="primary", size="lg")
                    
                    # 结果显示
                    output = gr.Textbox(
                        label="AI分析结果",
                        interactive=False,
                        lines=8
                    )
            
            # 为示例按钮绑定事件（在所有组件定义完成后）
            for btn, question in example_buttons:
                def make_click_handler(q):
                    def handler():
                        return q
                    return handler
                
                btn.click(
                    make_click_handler(question),
                    inputs=None,
                    outputs=question_input
                )
            
            # 提交问题
            submit_btn.click(
                fn=self.generate_response,
                inputs=[
                    image_input, 
                    question_input, 
                    max_tokens, 
                    temperature, 
                    min_p
                ],
                outputs=output
            )
            
            # 示例说明
            gr.Markdown(
                """
                ### 使用说明:
                1. **上传图片**: 点击上传按钮选择医学影像图片（支持JPG、PNG格式）
                2. **输入问题**: 在输入框输入您的问题，或点击示例问题
                3. **获取分析**: 点击"分析影像"按钮，AI放射科医生将分析影像并提供专业描述
                
                ### 注意事项:
                - 本系统仅供医学研究和教学使用，不能替代专业医疗诊断
                - 上传的图片仅用于本次分析，不会存储
                - 如遇紧急医疗情况，请立即联系专业医疗机构
                """
            )
        
        return interface

def main():
    """启动医学影像对话UI"""
    
    # 模型路径
    model_path = "./lora_model"  #微调后的模型
    # model_path = "./models/unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    # 创建UI实例
    try:
        medical_ui = SimpleMedicalUI(model_path)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 测试模型
    # print("🧪 测试模型...")
    # test_result = medical_ui.test_model()
    
    # if not test_result:
    #     print("❌ 模型测试失败，请检查模型")
    #     return
    
    # 创建界面
    interface = medical_ui.create_interface()
    
    # 启动服务
    print("🚀 启动医学影像AI助手...")
    print("📱 请在浏览器中访问: http://localhost:6006")
    print("⏹️ 按 Ctrl+C 停止服务")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=6008,
        share=False,
    )

if __name__ == "__main__":
    main()