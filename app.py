# app.py
# Main application file for the Gradio Chat Application with MCP Server

import gradio as gr
import requests
import json

# Configuration
MCP_SERVER_URL = 'http://your-mcp-server/api/chat' # Replace with actual URL if different

# Function to interact with MCP Server
def chat_with_mcp(user_message):
    # Sends a message to the MCP server and returns the reply.
    #
    # Args:
    # user_message (str): The message from the user.
    #
    # Returns:
    # str: The server's reply or an error message.

    payload = {"message": user_message}
    headers = {"Content-Type": "application/json"}

    try:
        # Send POST request to the MCP server
        response = requests.post(MCP_SERVER_URL, json=payload, headers=headers, timeout=10) # 10-second timeout
        response.raise_for_status()  # Raise an HTTPError for bad responses (4XX or 5XX)

        # Parse the JSON response
        data = response.json()

        # Extract the reply
        reply = data.get("reply")

        if reply is None:
            return "Error: 'reply' not found or is null in server response."

        return reply

    except requests.exceptions.Timeout:
        return f"Error: Request to MCP Server timed out. URL: {MCP_SERVER_URL}"
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to MCP Server. Check if the server is running and the URL is correct: {MCP_SERVER_URL}"
    except requests.exceptions.HTTPError as http_err:
        error_details = str(http_err)
        if hasattr(response, 'text') and response.text:
            error_details = f"{http_err}. Response: {response.text}"
        status_code = response.status_code if hasattr(response, 'status_code') else 'N/A'
        return f"Error: HTTP error occurred: {error_details}. Status code: {status_code}"
    except requests.exceptions.RequestException as req_err:
        return f"Error: An unexpected error occurred with the request: {req_err}"
    except json.JSONDecodeError:
        response_text = response.text if hasattr(response, 'text') else 'No response text available'
        return f"Error: Could not decode JSON response from server. Response text: {response_text}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {e}"

# Gradio Interface
iface = gr.Interface(
    fn=chat_with_mcp,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs=gr.Textbox(label="MCP Server Response"),
    title="MCP Chat Application",
    description="Enter your message and get a response from the MCP server. (MCP URL: http://your-mcp-server/api/chat)"
)

# Launch the application
if __name__ == '__main__':
    print("Attempting to launch Gradio app...")
    # For subtask environment, we don't actually launch as it might require a display or open ports.
    # iface.launch()
    print("Gradio app setup is complete. In a real environment, iface.launch() would start the server.")
    print(f"MCP Server URL is configured as: {MCP_SERVER_URL}")
