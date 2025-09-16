## Client X, an e-commerce clothing company wants to develop a new generative Al driven chat bot for their users to interact with on their homepage. The purpose of the chat bot will be to provide a way for the user to find products via a natural language interface.
Available to the development team will be datasets including customer purchase history and demographics, as well as a database of products with some information on product attributes. The attributes in this product database is of medium quality, with some values missing and some values being incorrectly populated.
Key concerns of the client are latency, scalability and cost. As part of your design, the client would also like you to build in a way to evaluate the success of the chatbot once it is in production.
Describe your thought process as you design and architect the solution. Ask the client any questions that will help you clarify exactly what they want to inform your design.
All data is currently available in the cloud in a Data Warehouse (e.g BigQuery, Redshift, Azure Synapse)
Business Requirements:

Latency: <2s for each chat message
Number of users: 5,000 unique sessions per hour
Budget per month: £5,000 per month
Success Metrics: User adoption
