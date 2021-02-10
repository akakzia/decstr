# Grounding Language to Autonomously-Acquired Skills via Goal Generation

This repository contains the code associated to the *Grounding Language to Autonomously-Acquired Skills via Goal Generation* paper submitted at the ICLR 2021 conference.

**Abstract**
We are interested in the autonomous acquisition of repertoires of skills. Language-conditioned reinforcement learning (LC-RL) approaches are great tools in this quest, as they allow to express abstract goals as sets of constraints on the states. However, most LC-RL agents are not autonomous and cannot learn without external instructions and feedback. Besides, their direct language condition cannot account for the goal-directed behavior of pre-verbal infants and strongly limits the expression of behavioral diversity for a given language input. To resolve these issues, we propose a new conceptual approach to language-conditioned RL: the Language-Goal-Behavior architecture (LGB). LGB decouples skill learning and language grounding via an intermediate semantic representation of the world. To showcase the properties of LGB, we present a specific implementation called DECSTR. DECSTR is an intrinsically motivated learning agent endowed with an innate semantic representation describing spatial relations between physical objects. In a first stage (G -> B), it freely explores its environment and targets self-generated semantic configurations. In a second stage (L -> G), it trains a language-conditioned  goal generator to generate semantic goals that match the constraints expressed in language-based inputs. We showcase the additional properties of LGB w.r.t. both an end-to-end LC-RL approach and a similar approach leveraging non-semantic, continuous intermediate representations. Intermediate semantic representations help satisfy language commands in a diversity of ways, enable strategy switching after a failure and facilitate language grounding.

**Link to Website**

You can find videos on the [paper's website](https://sites.google.com/view/decstr).

The paper can be found on Arxiv [here](https://arxiv.org/abs/2006.07185).


**Requirements**

* gym
* mujoco
* pytorch
* pandas
* matplotlib
* numpy

To reproduce the results, you need a machine with **24** cpus.

**Running a pre-trained policy**

Running the following line will trigger evaluation of a pre-trained agent. It will consecutively target each of the 35 valid configurations.

```python demo.py```

**Running a pre-trained policy with language**

You can run a pre-trained policy with a pre-trained language module. The language module is a generative model that samples a semantic goal configuration compatible with the 
instruction. Many configurations can satisfy a given instructions. The following command shows our the agent satisfies single instructions.

```python demo_sentence_test.py``` 

As the language module generates sets of compatible configuration, we can combine them to form any logical combinations of instructions. Try this with: 

```python demo_sentence_expr.py``` 


**Training DECSTR**

The following line trains phase 1: the sensorimotor learning part. Agents learn to explore their semantic representation space, to discover reachable configurations and to master 
them.

```mpirun -np 24 python train.py --env-name FetchManipulate3Objects-v0 --algo semantic```

**Training the Position Goals and Language Goals baselines**

The following lines train agents with the baselines conditions *Position Goals* and *Language Goals*. These are not expected to work (see paper).

```mpirun -np 24 python train.py --env-name FetchManipulate3ObjectsContinuous-v0 --algo continuous```

```mpirun -np 24 python train.py --env-name FetchManipulate3Objects-v0 --algo language```

**Training the Language module** 

The language-conditioned goal generation module can be trained with:

```python train_language_module_binary.py ```

The corresponding scripts can be found in the language folder. Once it is trained, you can test a trained agent with language instructions. See scripts demo_sentence_*
