from gradio_client import Client, handle_file


def edit_image_camera_angle(
    image_path: str,
    rotate_degrees: float = 0,
    move_forward: float = 0,
    vertical_tilt: float = 0,
    use_wide_angle: bool = False,
    prompt: str = "",
    seed: int = 0,
    randomize_seed: bool = True,
    guidance_scale: float = 1.5,
    num_inference_steps: int = 4,
) -> str:
    """Edit an image's camera angle using the Qwen Image Edit API.

    Args:
        image_path: Local file path or URL to the input image.
        rotate_degrees: Left/right rotation in degrees (-90 to 90).
        move_forward: Forward movement / zoom-in amount.
        vertical_tilt: Vertical tilt — negative = worm's-eye, positive = bird's-eye.
        use_wide_angle: Whether to apply a wide-angle lens effect.
        prompt: Optional text prompt for additional control.
        seed: Random seed value.
        randomize_seed: If True, a new seed is chosen on every run.
        guidance_scale: Classifier-free guidance scale.
        num_inference_steps: Number of denoising steps (4 for fast mode).

    Returns:
        Path or URL of the generated output image.
    """
    client = Client("linoyts/Qwen-Image-Edit-Angles")

    result = client.predict(
        image=handle_file(image_path),
        rotate_degrees=rotate_degrees,
        move_forward=move_forward,
        vertical_tilt=vertical_tilt,
        use_wide_angle=use_wide_angle,
        prompt=prompt,
        seed=seed,
        randomize_seed=randomize_seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        api_name="/predict",
    )

    return result


if __name__ == "__main__":
    result = edit_image_camera_angle(
        image_path="10Minutes_Smart.png",
        rotate_degrees=0,
        move_forward=0,
        vertical_tilt=0,
        use_wide_angle=False,
        prompt="",
        seed=0,
        randomize_seed=True,
        guidance_scale=1.5,
        num_inference_steps=4,
    )
    print(f"Generated Image result: {result}")
