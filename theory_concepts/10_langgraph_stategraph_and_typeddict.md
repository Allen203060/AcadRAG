# LangGraph: StateGraph & TypedDict Fundamentals

## 1. TypedDict (State Management)
LangGraph operates as a state machine. The "State" is the shared memory object passed between every node in the graph.
* **Definition:** We use Python's `TypedDict` to define a strict schema for this memory (e.g., `question`, `documents`, `answer`).
* **State Updates (The Merge):** Nodes do not overwrite the entire state. When a node returns `{"answer": "Hello"}`, LangGraph automatically merges this update into the global state, preserving all other existing keys.

## 2. StateGraph (The Orchestrator)
`StateGraph` orchestrates the flow of data using Graph Theory principles.
* **Nodes:** Python functions that perform discrete tasks (e.g., retrieving, grading, generating). They accept the `state` as input and return a dictionary of state updates.
* **Edges:** Define the routing logic. Standard edges enforce a linear progression (Node A $\rightarrow$ Node B).
* **START and END:** Virtual nodes that define the entry point of the user's initial state and the exit point where the graph halts execution.
* **Future Potential (Conditional Edges):** Because it is a graph, we can eventually add cyclical routing (e.g., if a hallucination is detected, route the state backward to the retrieval node for self-correction).
