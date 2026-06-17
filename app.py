import gradio as gr
from qwen_image_edit import edit_image_camera_angle


def run_edit(
    image,
    rotate_degrees,
    move_forward,
    vertical_tilt,
    use_wide_angle,
    prompt,
    seed,
    randomize_seed,
    guidance_scale,
    num_inference_steps,
):
    if image is None:
        raise gr.Error("이미지를 업로드해 주세요.")

    result = edit_image_camera_angle(
        image_path=image,
        rotate_degrees=rotate_degrees,
        move_forward=move_forward,
        vertical_tilt=vertical_tilt,
        use_wide_angle=use_wide_angle,
        prompt=prompt,
        seed=int(seed),
        randomize_seed=randomize_seed,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
    )

    # result may be a tuple (image_path, seed) or just a path string
    if isinstance(result, (list, tuple)):
        output_image = result[0]
        used_seed = result[1] if len(result) > 1 else seed
    else:
        output_image = result
        used_seed = seed

    return output_image, int(used_seed)


with gr.Blocks(title="Qwen 카메라 앵글 편집기") as demo:
    gr.Markdown(
        """
        # 📷 Qwen 카메라 앵글 편집기
        이미지의 카메라 앵글을 AI로 자유롭게 편집하세요.
        Powered by [linoyts/Qwen-Image-Edit-Angles](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="filepath", label="입력 이미지")

            with gr.Group():
                gr.Markdown("### 카메라 앵글 조정")
                rotate_degrees = gr.Slider(
                    minimum=-90, maximum=90, value=0, step=1,
                    label="좌우 회전 (°)"
                )
                move_forward = gr.Slider(
                    minimum=-1.0, maximum=1.0, value=0, step=0.05,
                    label="전진 / 클로즈업"
                )
                vertical_tilt = gr.Slider(
                    minimum=-1.0, maximum=1.0, value=0, step=0.05,
                    label="수직 틸트 (- 로우앵글 ↔ + 하이앵글)"
                )
                use_wide_angle = gr.Checkbox(
                    label="광각 렌즈 사용", value=False
                )

            with gr.Group():
                gr.Markdown("### 고급 설정")
                prompt = gr.Textbox(
                    label="추가 텍스트 프롬프트 (선택 사항)", placeholder="예: cinematic lighting"
                )
                with gr.Row():
                    seed = gr.Number(label="시드", value=0, precision=0)
                    randomize_seed = gr.Checkbox(label="시드 무작위화", value=True)
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=5.0, value=1.5, step=0.1,
                    label="가이드라인 스케일"
                )
                num_inference_steps = gr.Slider(
                    minimum=1, maximum=20, value=4, step=1,
                    label="추론 스텝 수"
                )

            submit_btn = gr.Button("생성하기", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="결과 이미지", interactive=False)
            used_seed = gr.Number(label="사용된 시드", interactive=False)

    submit_btn.click(
        fn=run_edit,
        inputs=[
            input_image,
            rotate_degrees,
            move_forward,
            vertical_tilt,
            use_wide_angle,
            prompt,
            seed,
            randomize_seed,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[output_image, used_seed],
    )

    gr.Examples(
        examples=[
            ["10Minutes_Smart.png", 15, 0, 0, False, "", 42, False, 1.5, 4],
            ["10Minutes_Smart.png", 0, 0.3, -0.4, False, "cinematic lighting", 0, True, 1.5, 4],
            ["10Minutes_Smart.png", -20, 0, 0.5, True, "", 0, True, 1.5, 4],
        ],
        inputs=[
            input_image, rotate_degrees, move_forward, vertical_tilt,
            use_wide_angle, prompt, seed, randomize_seed,
            guidance_scale, num_inference_steps,
        ],
        label="예시",
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
