---
layout: home
---

<article>
    
<!--add two gifs with captions one over other-->
    <h2>Adding New Feedback To Memory</h2>
    <div class="image fit">
        <img src="/assets/gifs/memprompt-populate.gif" alt="MemPrompt" />
        <figcaption><b>Memory is populated with new feedback: </b>A user enters a question for which no feedback is available (steps 1, 2).
            Directly prompting GPT-3 with the question leads to incorrect answer and understanding (step 3). User-provides
            feedback on the incorrect understanding (step 4), which is added to memory (step 5).</figcaption>
    </div>
    
    <hr>
    <h2>Utilizing Old Feedback To Answer New Questions</h2>
    
    <div class="image fit">
        <img src="/assets/gifs/memprompt-retrieve.gif" alt="MemPrompt" />
        <figcaption><b>Retrieving feedback from memory:</b> A user enters a question which GPT-3 has incorrectly
            answered in the past, and has received feedback from a user (step 1). The feedback is retrieved from memory (step
            2), and both question and feedback are added to the prompt. The prompt contains examples that allow GPT-3 to
            react to user feedback and generate correct understanding and answer.</figcaption>
    </div>

</article>