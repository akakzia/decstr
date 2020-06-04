# DECSTR: Learning Goal-Directed Abstract Behaviors using Pre-Verbal Spatial Predicates in Intrinsically Motivated Agents

This repository contains the code associated to the *DECSTR: Learning Goal-Directed Abstract Behaviors using Pre-Verbal Spatial Predicates in Intrinsically Motivated Agents* paper.

**Abstract**
Intrinsically motivated learning agents freely explore their environment and set their own goals. These goals are traditionally represented as specific configurations of the 
sensory inputs of the agent, but recent works introduced the use of language to represent more abstract goals. Language can, for example, represent goals as sets of general 
properties that surrounding objects should verify. However, language-conditioned agents are trained simultaneously to understand language and to act, which seems to contrast 
with how children learn: developmental psychology showed that infants demonstrate goal-oriented behaviors and abstract spatial concepts very early in their development, before 
language mastery. Guided by these findings, we introduce a high-level state representation based on natural semantic predicates that characterize spatial relations between 
objects and are known to be present early in infants. In a robotic manipulation environment, our DECSTR system explores this representation space by manipulating objects and 
efficiently learns to achieve any reachable goal configuration within it. It does so by leveraging an object-centered modular architecture, a symmetry inductive bias, and a new form of automatic curriculum learning for goal selection and policy learning. As with children, language acquisition takes place in a second phase, independently from goal-oriented sensorimotor learning. This is done via a new goal generation module, conditioned on instructions about transformations of object relations. We present ablations studies for each component and highlight several advantages of targeting abstract goals over specific ones. We further show that using this intermediate representation enables efficient language grounding by evaluating agents on sequences of language instructions and their logical combinations.

**Link to Website**

You can find videos on the [paper's website](https://sites.google.com/view/decstr).

The paper can be found on Arxiv [here]().


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

```python demo_sentece_test.py``` 

As the language module generates sets of compatible configuration, we can combine them to form any logical combinations of instructions. Try this with: 

```python demo_sentece_expr.py``` 


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



**TODO**
Add descriptions to run demos with language.