import gradio as gr
from transformers import pipeline
from PIL import Image
from functools import lru_cache

# Cache instantiated pipelines keyed by the model id.
@lru_cache(maxsize=3)
def get_model_pipeline(model_id: str):
    """
    Returns a text-generation pipeline from Hugging Face given the model_id.
    Caching is used to avoid frequent model reloads.
    """
    try:
        return pipeline("text-generation", model=model_id, max_new_tokens=150, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model {model_id}: {str(e)}")
        return None

def generate_response(user_message: str, image, chat_history, model_id: str):
    """
    Generates a response from the language model.
    
    Parameters:
      user_message (str): The text that the user typed.
      image: Optional image data uploaded by the user (as a numpy array).
      chat_history (list): A list of dictionaries with keys 'role' and 'content'.
      model_id (str): The Hugging Face model identifier to use.
      
    The function builds a conversational prompt by concatenating previous messages.
    If an image is provided, a brief note about it is appended to the message.
    The prompt is then sent to the generation pipeline and the generated response is extracted.
    """
    # If the user provided an image, add a note with its size.
    if image is not None:
        try:
            if not isinstance(image, Image.Image):
                img = Image.fromarray(image)
            else:
                img = image
            image_info = f"[Image uploaded (size: {img.size[0]}x{img.size[1]})]"
        except Exception:
            image_info = "[Image uploaded]"
        # Append image info to the user message.
        user_message = user_message + "\n" + image_info

    # Build the conversation prompt from the chat history.
    conversation = ""
    if chat_history:
        for msg in chat_history:
            if msg.get("role") == "user":
                conversation += f"User: {msg.get('content')}\n"
            elif msg.get("role") == "assistant":
                conversation += f"Assistant: {msg.get('content')}\n"
    conversation += f"User: {user_message}\nAssistant:"
    
    # Retrieve or load the generation pipeline for the chosen model.
    gen_pipeline = get_model_pipeline(model_id)
    if gen_pipeline is None:
        error_message = (
            f"Failed to load model: {model_id}. Please check if the model ID is correct "
            "and your internet connection. Some models may require specific configurations "
            "or might not support direct text generation."
        )
        if chat_history is None:
            chat_history = []
        new_history = chat_history.copy()
        new_history.append({"role": "user", "content": user_message})
        new_history.append({"role": "assistant", "content": error_message})
        return new_history, ""

    try:
        # Generate response text.
        output = gen_pipeline(conversation)[0]["generated_text"]
        # Extract the new portion of the text.
        bot_response = output[len(conversation):].strip()
    except Exception as e:
        bot_response = f"Error generating response: {str(e)}"
    
    # Update the conversation history using OpenAI-style message dictionaries.
    if chat_history is None:
        chat_history = []
    new_history = chat_history.copy()
    new_history.append({"role": "user", "content": user_message})
    new_history.append({"role": "assistant", "content": bot_response})
    return new_history, ""

# CSS string for a clean, modern, and attractive look.
css_str = """
/* Global and container styles */
body { 
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
    background-color: #f5f5f5; 
}
.gradio-container { 
    max-width: 800px; 
    margin: auto; 
}

/* Chatbot styling */
.gradio-chatbot { 
    border-radius: 8px; 
    background: #ffffff; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    padding: 10px;
}
"""

# Build the Gradio Blocks app interface.
with gr.Blocks(css=css_str) as demo:
    gr.Markdown("# Hugging Face Chat with Image Upload")
    gr.Markdown(
        "Interact with a Hugging Face LLM in a modern chat interface. "
        "Type your message and optionally upload an image—the model will consider both. "
        "**Note:** Some models might require `trust_remote_code=True` to load properly. "
        "If you encounter errors, try switching to a different model."
    )
    
    # Model selection with improved options
    with gr.Row():
        model_input = gr.Dropdown(
            label="Select Hugging Face Model",
            choices=[
                "LGAI-EXAONE/EXAONE-Deep-2.4B",
                "ds4sd/SmolDocling-256M-preview",
                "google/flan-t5-small"
            ],
            value="LGAI-EXAONE/EXAONE-Deep-2.4B",
            info="EXAONE-Deep-2.4B is recommended for better text generation"
        )
    
    # Chat conversation history using the new 'messages' type.
    chatbot = gr.Chatbot(label="Chat", type="messages")
    
    # Input area with text and optional image upload.
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(
                label="Your Message", 
                placeholder="Type your message here...",
                lines=2
            )
        with gr.Column(scale=2):
            # Removed the unsupported 'tool' argument.
            image_input = gr.Image(
                label="Upload Image (Optional)", 
                type="numpy"
            )
    
    # Action buttons.
    with gr.Row():
        send_button = gr.Button("Send")
        clear_button = gr.Button("Clear Chat")
    
    # Define actions for buttons.
    send_button.click(
        generate_response, 
        inputs=[user_input, image_input, chatbot, model_input],
        outputs=[chatbot, user_input]
    )
    
    clear_button.click(
        lambda: ([], ""), 
        None, 
        [chatbot, user_input]
    )

# Launch the Gradio app.
demo.launch()
