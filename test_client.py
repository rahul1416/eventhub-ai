import asyncio
import websockets

async def test_chat():
    uri = "ws://localhost:8000/ws/chat"

    async with websockets.connect(uri) as websocket:
        # Simulate sending messages
        messages = [
            "Hey, did anyone watch the keynote?",
            "Yes! The AI section was amazing!",
            "I liked the demo on real-time voice translation.",
            "What was the name of the presenter again?",
            "I think it was Dr. Lisa Wong.",
            "Yeah, and she mentioned something about a new open-source project.",
            "Exactly, it’s called WhisperX!"
        ]

        async def send_messages():
            for msg in messages:
                await websocket.send(msg)
                print(f"Sent: {msg}")
                await asyncio.sleep(2)

        async def receive_messages():
            try:
                while True:
                    response = await websocket.recv()
                    print(f"Received: {response}")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")

        await asyncio.gather(send_messages(), receive_messages())

# Run the test
asyncio.run(test_chat())
