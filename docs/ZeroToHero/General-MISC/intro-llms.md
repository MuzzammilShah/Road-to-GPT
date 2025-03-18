# **INTODUCTION TO LLMs**

--------

!!! warning "Development in Progress"

    **Notes for this is ongoing**

## LLM Inference

A LLM is essentially two files (assuming we have a hypothetical directory) where there is a parameters file and a code file (c or python, any arbitrary language).

The model in the hypothetical directory we are assuming is llama2 70b. This is popular because it is a open weights model where the weights, architecture and the paper were all released by Meta. Unlike ChatGPT, Claude or Grok were the model architecture was never released and is owned by the respective companies. We only use those models through a web interface but we actually don’t have access to that model.

So coming back to our directory: we have a parameters file and a code file which runs those parameters. The parameters are the weights/parameters of the neural net which is actually the language model.

Now, because this is a 70b parameter model, each parameter is stored as 2 bytes so the total file size will be of 140GB. Its 2 bytes because this is a float16 number i.e. it's data type. This file just contains a large number/lists of parameters for that neural net. And we need something to run that, that where the code file comes in.

It wouldn't take more than 500 lines of code (lets say in C) with no other dependencies to implement this neural network architecture, which in turn uses those parameters to run the model.

These two files form a package of their own, so you don’t need anything else to run this model (not even the internet), so you just compile the C and it generates the binaries to point to the parameters and you can start to talk to this model.

Also, larger the parameter size of the model, more time it takes to run.

Now, running the model is not really a big task, not a lot is required. But the computational complexity comes in when we would like to get/generate those parameters. So how do we get the parameters? where are they from?

Whatever is in the code file i.e. the neural network architecture - the forward pass implementation etc., is all algorithmically understood and so on. But the magic really happens in the parameters. Therefore, how do we obtain them?

&nbsp;

## LLM Training

The parameters are obtained through a method called Training. So this step is essentially called 'Model Training'. What we read about before this is called 'Model Inference' where we just run the final model on a Mac or any device of yours. 

This process is where there are more computational effort is needed. So we take a large chunk of the internet ~10TB of data (this is obtaining by crawlers) -> this is then passed to a cluster of GPUs (6000 GPUs running for 12 days costing about 2 million dollars) -> finally to obtain the parameters file.

Now the final parameters file can be thought of a zip file, except incase of a zip file its lossless compression, here there is loss. So what is happened is that there is a gibberish bulk of data (LLM "dreams" the internet data, where the texts produced wont necessarily make sense and it just generates something similar to what it learnt. Like codes, product reviews, articles etc.) that is actually generated based on the chunk of data provided to it, its not really a replica of it. Also notice that the file size is also considerably less as compared to the input chunk of data 10TB -> 140GB.

Just so we know, the above estimates of values in today's terms are rookie numbers. If you want to obtain models by OpenAI, Claude etc., then you just 10x those values. Therefore, they even cost tens of millions of dollars.

Now, its essential we talk about what a neural network actually is. The most popular explanation is ofcourse the 'predicting the next word' example where the NN is fed in some sentence and it is supposed to predict the next word in the sentence/finish it. 

This may seem like a fairly simple task, but this is one of the best possible assignment for a NN to learn from. Now consider this example, the NN is fed a page from wikipedia on 'Naruto'. It needs to learn everything about Naruto inorder to answer/complete the sentence on it. For example, who are his parents, name of his village, name of his sensei etc. Since it has been trained on all of this, and the user types in the sentence: `Naruto wants to become a _`, the model would know what the next word is and complete it by saying `Hokage` with probably a ~98% match.

***Fun fact: Till here is where the [GPT-1 implementation](https://docs.muhammedshah.com/ZeroToHero/GPT-1/) was done in the learning path :)***

&nbsp;

## How do they work?

Here is where we drop probably the most famous, groundbreaking concept in the world of ML - The Transformer architecture. The best part about this is that we know each and every part of this structure and what each of those sections perform mathematically. 

But little is known in full detail:

- The billions of parameters are just dispersed through the network of this system.
- We know how to iteratively adjust or optimize these parameters over time to make it better at prediction.
- We don't really know how the billions of parameters collaborate to do it.

So ultimately we can just measure how we can improve them but we have no idea how/what it's actually doing to get there.

What we know is that these NN build and maintain some kind of knowledge database to store all the information learnt, but even the data there is bit strange and imperfect. For example, there is this example called "reversal curse" where:

```
Q. Who is Naruto's father?
A. Minato, the fourth hokage.
```

But if you ask,

```
Q. Who is Minato?
A. I have no idea.
```

So ultimately, you can see LLMs as a inscrutable artifacts. What we mean by that is it cannot be studied like a car where we know all the different components. Although there are some advanced fields where researchers go hands on and dig deep to understand how every section work but yes it is a very sophisticated process.

*I am thinking maybe we can bring in one of chris nolan's quote from TENET here: "Don't try to understand it, feel it." (Not cool for academic situations I know, but I couldn't help myself lol)*

&nbsp;

## Finetuning into an Assistant

The above process was just the first step of training called the 'pre-training'. Now comes the second step in training called 'finetuning'.

The documents obtained through pre-training is not very useful as its just random chunk of data, so we need to turn that into an assistant like model.

How we do that is, we swap the dataset (internet data) on which the training was happening and replace it with the data which is a lot more formatted, i.e. in a question and answer format. Which is either written by a large group of people or a model/AI itself generates it (synthetic data).

Therefore is you look at it, in the first step it was quantity over quality but in the second step it is vice versa.

Now also note that, after having done the fine tuning, if the user asks something which wasn't in the training, the model will still answer as it was trained to be a helpful assistant and must answer in that way. So its still generating a sequence of data but in a more structured/fine tuned way - left to right and then right to bottom when it comes to framing words -> sentences -> paragraphs. In the end you can say its more of a formatting stage as well as it is converting internet data into a more structured helpful assistant like manner.

&nbsp;

## Summary so far

&nbsp;

<div style="text-align: center;">
    <img src="../../../assets/images/into-to-llm-summary.png" alt="Summary image">
</div>

&nbsp;

Now, the llama2 model that we took as example, Meta had released both base model and assistant model. The base model ofcourse aren't very useful, so if you ask it a question it will probably generate more questions. So what Meta has done is that they did the tough part and provided us with the base model so that we can fine tune it ourselves (and they have also directly provided the assistant model so we can directly use it).

Other ways of fine tuning mentioned in Appendix chapter of the video:

- Comparisons: providing different options
- Labeling docs: an instruction document is provided as to how you need to label the data
- RLHF: Reinforcement Learning with Human Feedback (the name prettymuch explains it lol)
- Leaderboard: Like a chatbot arena, to rate which model performs better

&nbsp;

## INTERLUDE

So, I have skipped the notes for Part 2 as at the time of watching the video vs the time it was recorded by sensei, a lot has changed, infact a lot of his ideas were also implemented (also goes on to show how brilliant this man actually is)

If we take cases of some of his points like:

- In the 'Leaderboard' we see how only closed models like ChatGPT and Claude are on the top, only to be followed by open source models later behind. We couldn't finetune or modify those top performing models as they are closed source -> But now we had [Deepseek](https://www.deepseek.com/) who entered the game, top the charts with the performance and even open sourced the model and is now ahead.
- 'Tool Use' -> Now known as 'AI Agents' where LLMs are used as the Brain of a system and tools are provided to it to perform various other tasks. Ofcourse one example was already there at the time in the implementation of DALL-E, the image generation model.
- 'Thinking Systems' -> Again Deepseek comes into picture here, but with OpenAI's o1 model which was released slightly earlier, where the model would be provided time to think to frame its answers and perform research.
- 'LLM OS' -> This is something which is not yet achieved at the time of me writing this, but i do know some companies which are on route to achieving this. My personal favorite that I am keeping track of is [Letta AI](https://www.letta.com/), where i first learnt about this :)

&nbsp;