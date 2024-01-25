
# TODO: serve and receive websocket messages

# https://towardsdatascience.com/build-an-ai-based-autocomplete-in-the-browser-using-vue-js-fastapi-and-websockets-1eb7ae19bfd8
# https://medium.com/mlearning-ai/object-detection-service-with-yolo-and-fastapi-af1318ee73ed
# https://github.com/wingedrasengan927/deep-autocomplete
# https://github.com/c0sogi/LLMChat
# https://fastapi.tiangolo.com/project-generation/#machine-learning-models-with-spacy-and-fastapi
# https://fastapi.tiangolo.com/features/
# https://fastapi.tiangolo.com/advanced/websockets/
# https://developer.mozilla.org/es/docs/Web/API/WebSockets_API

# Real-time Object Detection and Classification with FastAPI, PyTorch, and GCP
# Interactive Chatbot with PyTorch and FastAPI
# Image Segmentation Service

"""
Websockets provide a communication protocol that enables bidirectional, real-time communication between a client (such as a web browser) and a server over a single, long-lived connection. 
Unlike the traditional request-response model of HTTP, where the client sends a request and the server responds, websockets allow data to be sent and received asynchronously at any time.

Here are some key characteristics and features of websockets:

Bidirectional Communication:
    Websockets allow data to be sent in both directions, from the client to the server and vice versa. 
    This bidirectional communication is crucial for real-time applications.
Full-duplex Communication:
    Websockets enable full-duplex communication, meaning that data can be sent and received simultaneously. 
    This is in contrast to the half-duplex nature of traditional HTTP, where communication occurs in one direction at a time.
Persistent Connection:
    Once a websocket connection is established, it remains open for the duration of the communication session. 
    This persistence eliminates the need to repeatedly open and close connections for each piece of data.
Low Latency:
    Websockets are designed to reduce latency and overhead compared to traditional HTTP. 
    This makes them suitable for applications that require real-time updates and responsiveness.
Efficient for Streaming:
    Websockets are well-suited for streaming data scenarios, such as live updates, real-time notifications, and continuous feeds.
Protocol Upgrade:
    Websockets are initiated with an HTTP request, and if both the client and server support websockets, the connection can be upgraded from HTTP to the websocket protocol.
WebSocket API:
    Web browsers provide a JavaScript API called the WebSocket API, which allows developers to work with websockets in the client-side code.
Security Considerations:
    Websockets can be used over secure connections (wss://), providing encryption for the data transmitted between the client and the server.
Cross-Origin Resource Sharing (CORS):
    Websockets adhere to CORS policies, and servers need to include appropriate headers to allow cross-origin communication.
Common Use Cases:

Websockets are commonly used in applications that require real-time features, such as chat applications, online gaming, live notifications, financial trading platforms, collaborative editing tools, and more.
In summary, websockets offer a more efficient and responsive way to handle real-time communication between clients and servers, making them suitable for a variety of interactive and dynamic web applications.
They are particularly valuable in scenarios where low latency and bidirectional communication are essential.

"""