# Concept 15: Open Source Software Licenses & Repository Governance

## 1. Why Open Source Licensing Matters
An open-source license explicitly communicates how other developers, researchers, and enterprises can legally use, modify, share, or monetize your software. Without a `LICENSE` file, copyright law defaults to "all rights reserved," meaning others cannot legally copy or use your code, even if it's hosted publicly on GitHub.

## 2. Permissive vs. Copyleft Licenses

### Category A: Permissive Licenses (Maximum Freedom)
1. **MIT License (Recommended for Portfolio/RAG Projects):**
   * **Summary:** Extremely permissive, short (1 page), and universally standard.
   * **Permissions:** Commercial use, modification, distribution, private use.
   * **Conditions:** Include original copyright notice.
   * **Liability:** Author holds zero warranty or legal liability.
2. **Apache 2.0:**
   * **Summary:** Permissive like MIT, but includes explicit patent rights protection and trademark disclaimers.

### Category B: Copyleft Licenses (Reciprocal Sharing)
1. **GNU GPLv3:**
   * **Summary:** Strong copyleft. Any derivative work or project using GPLv3 code must also be open-sourced under GPLv3.
2. **AGPLv3 (Affero GPL):**
   * **Summary:** Network copyleft. Extends GPL to cloud/SaaS implementations—if hosted as a web service, the source code must be made public.

## 3. Why MIT License is Best for AcadRAG
For an AI RAG engineering project meant for GitHub portfolio visibility, technical interviews, and community adoption:
* **Recruiter & Community Friendly:** Anyone can clone, test, or integrate parts of AcadRAG without legal friction.
* **Standard Practice:** Almost all major AI libraries (LangChain, PyTorch, Transformers) use permissive licenses (MIT or Apache 2.0).
