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
    return pipeline("text-generation", model=model_id, max_new_tokens=150)

def generate_response(user_message: str, image, chat_history, model_id: str):
    """
    Generates a response from the language model.
    
    Parameters:
      user_message (str): The text that the user typed.
      image: Optional image data uploaded by the user (as a numpy array).
      chat_history (list): A list of (user, assistant) tuples.
      model_id (str): The Hugging Face model identifier to use.
      
    The function builds a conversational prompt by concatenating the chat history.
    If an image is provided, a brief note about it is appended to the message.
    Then, the function calls the text-generation pipeline and extracts the newly generated text.
    """
    # If the user provided an image, add a note to indicate it.
    if image is not None:
        try:
            # Convert the numpy array to a PIL image (if it isn’t already)
            if not isinstance(image, Image.Image):
                img = Image.fromarray(image)
            else:
                img = image
            # You can expand this later with more image processing (e.g., OCR or image captioning)
            image_info = f"[Image uploaded (size: {img.size[0]}x{img.size[1]})]"
        except Exception:
            image_info = "[Image uploaded]"
        # Append image info to the user message.
        user_message = user_message + "\n" + image_info

    # Build the full conversation prompt.
    conversation = ""
    if chat_history:
        for (usr, bot) in chat_history:
            conversation += f"User: {usr}\nAssistant: {bot}\n"
    conversation += f"User: {user_message}\nAssistant:"

    # Retrieve or load the generation pipeline for the chosen model.
    gen_pipeline = get_model_pipeline(model_id)
    # Generate response text. Note that the model may include the prompt in its output.
    output = gen_pipeline(conversation)[0]["generated_text"]
    # Remove the conversation prompt so that only the new text remains.
    bot_response = output[len(conversation):].strip()
    
    # Update the conversation history and clear the text input after submission.
    if chat_history is None:
        chat_history = []
    chat_history.append((user_message, bot_response))
    return chat_history, ""

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
        "You can type your message and optionally upload an image—the model will consider both. "
        "Specify any Hugging Face model; the default is `ds4sd/SmolDocling-256M-preview`."
    )
    
    # Model selection: Allow the user to enter a Hugging Face model ID.
    with gr.Row():
        model_input = gr.Textbox(
            label="Hugging Face Model ID", 
            value="ds4sd/SmolDocling-256M-preview",
            interactive=True
        )
    
    # Chat conversation history. Gradio automatically formats the list of (user, assistant) tuples.
    chatbot = gr.Chatbot(label="Chat")
    
    # Use a row to place text and image input side by side.
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(
                label="Your Message", 
                placeholder="Type your message here...",
                lines=2
            )
        with gr.Column(scale=2):
            image_input = gr.Image(
                label="Upload Image (Optional)", 
                type="numpy", 
                tool="editor"
            )
    
    # Row for action buttons.
    with gr.Row():
        send_button = gr.Button("Send")
        clear_button = gr.Button("Clear Chat")
    
    # When the Send button is clicked, the generate_response function is called.
    send_button.click(
        generate_response, 
        inputs=[user_input, image_input, chatbot, model_input],
        outputs=[chatbot, user_input]
    )
    
    # Clear chat history when the Clear Chat button is clicked.
    clear_button.click(
        lambda: ([], ""), 
        None, 
        [chatbot, user_input]
    )

# Launch the Gradio app.
demo.launch()
