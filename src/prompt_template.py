no_pronoun_binary_choice_prompt_template = """
In the following sentence, who is more likely to be the subject of the next sentence? {} or {}? Please ONLY return the name without any explanation or extra words.
Sentence: {}
Answer:
"""

no_pronoun_yes_no_prompt_template = """
In the following sentence, judge whether the pronoun of the next sentence will refer to {}. Please ONLY answer with 'Yes' or 'No'.
Sentence: {}
"""

no_pronoun_continuation_prompt_template = """
Please reasonably continue the sentence with either of the mentioned characters. You should start a new sentence rather than a clause. Please ONLY return the continuation.
Sentence: {}
"""

pronoun_binary_choice_prompt_template = """
In the following sentence, who is more likely to be the referent of the pronoun? {} or {}? Please ONLY return the name without any explanation or extra words.
Sentence: {}
Answer:
"""

pronoun_yes_no_prompt_template = """
In the following sentence, judge whether the pronoun refers to {}. Please ONLY answer with 'Yes' or 'No'.
Sentence: {}
"""

pronoun_continuation_prompt_template = """
Please reasonably continue the sentence following the pronoun. Please ONLY return the continuation.
Sentence: {}
"""
