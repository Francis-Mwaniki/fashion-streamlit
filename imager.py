import replicate
import streamlit as st
import requests
import zipfile
import io
from streamlit_image_select import image_select
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()


# API Tokens and endpoints from `.streamlit/secrets.toml` file
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_MODEL_ENDPOINTSTABILITY = os.getenv("REPLICATE_MODEL_ENDPOINTSTABILITY")

# Resources text, link, and logo
replicate_text = "Stability AI SDXL Model on Replicate"
replicate_link = "https://replicate.com/stability-ai/sdxl"
replicate_logo = "https://storage.googleapis.com/llama2_release/Screen%20Shot%202023-07-21%20at%2012.34.05%20PM.png"

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()
def main_page(
              prompt: str) -> None:
    with st.container():
        with st.status('👩🏾‍🍳 Whipping up your words into art...', expanded=True) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and strecth in the meantime")
            try:
                # Only call the API if the "Submit" button was pressed
                
                with generated_images_placeholder.container():
                        all_images = []  # List to store all generated images
                        output = replicate.run(
                            REPLICATE_MODEL_ENDPOINTSTABILITY,
                            input={
                                "prompt": prompt,
                                "width": 640,  # "width": "width",
                                "height": 400,  # "height": "height",
                                "num_outputs":1,  # "num_outputs": "num_outputs",
                                "scheduler": "DDIM",  # "scheduler": "scheduler",
                                "num_inference_steps": 50,  # "num_inference_steps": "num_inference_steps",
                                "guidance_scale": 7.5,  # "guidance_scale": "guidance_scale",
                                "prompt_stregth": 0.8,  # "prompt_stregth": "prompt_stregth",
                                "refine":  "expert_ensemble_refiner",  # "refine": "refine",
                                "high_noise_frac": 0.8,  # "high_noise_frac": "high_noise_frac",
                            }
                        )
                        if output:
                            st.toast(
                                'Your image has been generated!', icon='😍')
                            # Save generated image to session state
                            st.session_state.generated_image = output

                            # Displaying the image
                            for image in st.session_state.generated_image:
                                with st.container():
                                    st.image(image, caption="Generated Image 🎈",
                                             use_column_width=True)
                                    # Add image to the list
                                    all_images.append(image)

                                    response = requests.get(image)
                        # Save all generated images to session state
                        st.session_state.all_images = all_images

                        # Create a BytesIO object
                        zip_io = io.BytesIO()

                        # Download option for each image
                        with zipfile.ZipFile(zip_io, 'w') as zipf:
                            for i, image in enumerate(st.session_state.all_images):
                                response = requests.get(image)
                                if response.status_code == 200:
                                    image_data = response.content
                                    # Write each image to the zip file with a name
                                    zipf.writestr(
                                        f"output_file_{i+1}.png", image_data)
                                else:
                                    st.error(
                                        f"Failed to fetch image {i+1} from {image}. Error code: {response.status_code}", icon="🚨")
                        # Create a download button for the zip file
                        st.download_button(
                            ":red[**Download All Images**]", data=zip_io.getvalue(), file_name="output_files.zip", mime="application/zip", use_container_width=True)
                status.update(label="✅ Images generated!",
                              state="complete", expanded=False)
            except Exception as e:
                print(e)
                st.error(f'Encountered an error: {e}', icon="🚨")



# def main():
#     # Set the page title and favicon
#     # Set the sidebar title

#     # Display the main page
#     main_page(
#         prompt=st.text_input("Prompt", "An astronaut riding a rainbow unicorn, cinematic, dramatic")
#     )

# if __name__ == "__main__":
#     main()