from agents.hypothesis_agent_rag import HypothesisAgentWithRAG

# IMPORTANT: Configurez votre email PubMed
agent = HypothesisAgentWithRAG(verbose=True)

symptoms = ["fever", "cough", "fatigue"]
hypotheses = agent.generate_hypotheses(symptoms, top_k=5)

print("\n📋 RAG-Enhanced Hypotheses:")
for i, hyp in enumerate(hypotheses, 1):
    print(f"{i}. {hyp['diagnosis']} (conf: {hyp['confidence']:.2f})")
    print(f"   Explanation: {hyp['explanation']}")
    print(f"   RAG-enhanced: {hyp['rag_enhanced']}")