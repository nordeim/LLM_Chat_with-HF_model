import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def chat_with_model(message, history, model_name, image_file):
    """
    Chat function that interacts with a Hugging Face language model.

    Args:
        message (str): User input text message.
        history (list): Chat history.
        model_name (str): Name of the Hugging Face model to use.
        image_file (TemporaryFile): Uploaded image file (can be None).

    Returns:
        tuple: Updated chat history and the model's response.
    """
    history = history or []

    if not message and not image_file:
        history.append((None, "Please provide text input, image input or both."))
        return history, history

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        error_message = f"Error loading model: {e}. Please check the model name and your internet connection."
        history.append((message, error_message)) # Append user message and error
        return history, history  # Return history to update chatbot with error

    user_message = message
    if image_file:
        image_message = f"<image: {image_file.name}>" # Indicate image upload in chat history - further image processing would be model specific
        user_message = f"{user_message} {image_message}" if message else image_message # Combine text and image info

    history.append((user_message, None)) # Append user message, model response is None initially

    # Prepare input for the model
    inputs = tokenizer(user_message, return_tensors="pt").to(model.device) # Ensure input is on the same device as model
    model = model.to("cpu") # Move model to CPU if GPU is not available or for testing (can be adjusted for GPU if needed)
    inputs = inputs.to("cpu") # Move inputs to CPU


    try:
        # Generate response
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model.generate(**inputs, max_length=500, num_return_sequences=1) # Adjust max_length as needed
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Basic cleanup for better readability - can be model specific and improved
        response = response.replace(user_message, "").strip() # Remove input echo if model does that
        if not response:
            response = "Sorry, I couldn't generate a meaningful response." # Handle empty response

    except Exception as e:
        response = f"Error generating response: {e}"

    history[-1] = (user_message, response) # Update last history item with model response
    return history, history


if __name__ == "__main__":
    default_model = "ds4sd/SmolDocling-256M-preview"
    available_models = [default_model, "gpt2", "google/flan-t5-small"] # Add more models as needed

    with gr.Blocks(title="Image & Text Chat with Hugging Face Model", theme=gr.themes.Soft()) as iface: # Use Blocks for more layout control and Soft theme
        gr.Markdown("# Let's Chat with Language Models!")
        gr.Markdown("Choose a model and start chatting. You can also upload an image along with your text.")

        model_selector = gr.Dropdown(available_models, value=default_model, label="Choose Model")

        chatbot = gr.Chatbot(height=400, show_copy_button=True) # Customize chatbot appearance - Removed bubble_radius
        msg = gr.Textbox(label="Your message", placeholder="Type your message here and press Enter", container=False)
        image_input = gr.File(label="Upload Image (Optional)", file_types=["image"]) # Use gr.File for file upload, or gr.Image for image display

        clear_btn = gr.ClearButton([msg, chatbot, image_input]) # Clear button for all relevant components

        def respond(message, chat_history, model_name, image_file): # Gradio expects separate function for event handling
            updated_history, updated_chatbot = chat_with_model(message, chat_history, model_name, image_file)
            return "", updated_history # Clear input textbox after submission, return updated history

        msg.submit(respond, [msg, chatbot, model_selector, image_input], [msg, chatbot]) # Submit on Enter key press

        gr.Examples( # Add example prompts
            examples=[
                ["Hello, how are you?", None, default_model],
                ["Tell me a story.", None, default_model],
                ["What can you do?", None, default_model],
                ["Describe this image.", None, default_model] # Example with image - changed image path to None
            ],
            inputs=[msg, image_input, model_selector],
            outputs=[chatbot],
            fn=respond,
            cache_examples=False, # Disable example caching for dynamic behavior if needed
        )
        gr.Markdown("Note: Image upload is included for potential future enhancements and context. Ensure the chosen model is capable of processing image information effectively. Currently, the application primarily focuses on text-based interaction with the selected language model.")
        gr.Markdown("Created with Gradio and Hugging Face Transformers.")

    iface.launch(share=False) # share=False for local execution, share=True to create a shareable link (be mindful of model resource usage)
