---
layout: home
---
<article>

MemPrompt is a general technique that can be used to <i>patch</i> GPT-3 after deployment.
To adapt MemPrompt for a task, we need to create prompts that include the following two components: (i) generating an understanding of the task ($u$), and (ii) reacting to user feedback ($fb$). Please see Section 3 of the <a href="https://arxiv.org/abs/2201.06009">paper</a> for more details.


<span style="background-color:#aeeffa;"></span>
<h2>Generating Task Understanding with Response</h2>

<p>Existing methods for receiving user feedback typically assume the user knows the correct answer.
This assumption is paradoxical: <b>if the user knew the answer, why would they be using the model?</b></p>

In real settings, the user may not know the correct answer, but may know what instruction they gave.
Thus, MemPrompt generates task understanding in response in addition to the answer. We operationalize this idea by including task verbalization in the prompt.

<p> Given a question ($x$) <i>What sounds like < sighted > ?</i>, a simple prompting approach will generate the answer ($y$) <i>cited</i> (say).
In contrast, we prompt the model to generate a task description <span style="background-color:#aeeffa;"><i>the homophone for</i></span>.</p>

To see why this is helpful, consider a test question <span style="background-color:#ffff0f;"><i>What sounds similar to < sighted > ?</i></span>. If the model generates <span style="background-color:#aeeffa;"><i>the word that has the same meaning</i></span> as its understanding of task, the user has a reason to believe that the answer is wrong.

<center>
<div class="image fit">
    <img src="/assets/images/verbalizing_understanding.jpg" alt="MemPrompt" />
</div>
</center>

<h2>Allowing GPT-3 to react to feedback</h2>
Once the feedback is received from the user, can the model successfully utilize it? By adding a few examples of the form $x, fb \rightarrow u, y$ in the prompt and setting $fb=u$, we force the model to use the task understanding present in the input when generating the output~(Figure below). 

<center>
<div class="image fit">
    <img src="/assets/images/reacting_to_feedback.jpg" alt="MemPrompt" />
</div>
</center>

</article>
    