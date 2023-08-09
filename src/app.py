import chainlit as cl

@cl.on_chat_start
def setup():
    pass


@cl.on_message
async def main(message: str):
    pass