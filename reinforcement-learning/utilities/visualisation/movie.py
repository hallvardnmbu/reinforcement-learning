import cv2
import torch


def create_movie(environment, agent, path, fps=60):
    """Created by Mistral Large."""
    initial = agent.preprocess(environment.reset()[0])
    try:
        states = torch.cat([initial] * agent.shape["reshape"][1], dim=1)
    except AttributeError:
        states = initial

    try:
        done = False

        # Get the dimensions of the first image
        height, width, channels = environment.render().shape

        # Create the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # You can change the codec if needed
        video_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        while not done:
            _, states, _, done = agent.observe(environment, states)
            video_writer.write(environment.render())
    except Exception as e:
        print(f"Error during image generation or writing: {e}")
        return

    cv2.destroyAllWindows()
