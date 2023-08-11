# researchGPT

researchGPT is a conversational agent designed to engage in high-level discussions about research topics. Leveraging the power of AI, researchGPT can locate relevant research papers, extract and embed their text, and autonomously generate responses based on the research content or perform web searches if needed. The application harnesses the capabilities of OpenAI's GPT-3.5 to comprehend human input and deliver meaningful responses and actions.

## Technologies

researchGPT is built on a stack of cutting-edge technologies:

- **OpenAI's GPT-3.5:** The core AI model that powers researchGPT's understanding of natural language and generates responses.
- **Semantic Scholar Database:** researchGPT taps into the Semantic Scholar database to source research papers related to user queries.
- **Langchain Framework:** The Langchain framework is employed to manage and process research paper data, allowing for efficient extraction and organization of text content.
- **Chainlit UI:** researchGPT utilizes Chainlit, providing an abstracted user interface for rapid deployment and interaction with the conversational agent.

## Features

- Seamless conversation with a conversational agent about high-level research topics.
- Automated retrieval and embedding of research paper content from the Semantic Scholar database.
- Autonomous responses based on the embedded research text or web searches when necessary.
- Leveraging OpenAI's GPT-3.5 for understanding and generating natural language responses.
- User-friendly interface powered by the Chainlit UI framework.

## Getting Started

To get started with researchGPT, follow these steps:

1. Clone this repository.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Configure your OpenAI API credentials and other settings.
4. Run the application and engage in insightful discussions about research! `chainlit run src/app.py`

## Usage

1. Launch the researchGPT application.
2. Initiate a conversation by providing a research topic or query.
3. researchGPT will respond with relevant information and engage in a meaningful discussion.

## Acknowledgements

This project is made possible by the contributions of the open-source community, the advancements in AI technology from OpenAI, and the convenience of the Langchain and Chainlit frameworks.

## License

This project is licensed under the [MIT License](LICENSE).
