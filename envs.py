import numpy as np
from sklearn.base import clone
import collections
from sklearn.ensemble import RandomForestClassifier
import random



class LalEnv(object):

    def __init__(self, dataset, model, quality_method):

        # Inits environment with attributes: dataset, model, quality function and other attributes.
        print("LalEnv init")
        print("\n")
        self.dataset = dataset
        print("self.dataset")
        print(self.dataset)
        print("\n")        
        self.model = model
        print("self.model")
        print(self.model)
        print("\n")    
        self.quality_method = quality_method
        print("self.quality_method")
        print(self.quality_method)
        print("\n")    

        # Compute the number of classes as a number of unique labels in train dataset:
        self.n_classes = np.size(np.unique(self.dataset.train_labels))

        # Initialize a list where quality at each iteration will be written:
        self.episode_qualities = []

        # Rewards bank to store the rewards.
        self.rewards_bank = []

    
    
    def reset(self, n_start=2):

        print("envs, def reset(self, n_start=2):")

        # SAMPLE INITIAL DATAPOINTS.
        self.dataset.regenerate()
        self.episode_qualities = []
        self.episode_qualities.append(0)
        self.rewards_bank = []
        self.rewards_bank.append(0)

        # To train an initial classifier we need at least self.n_classes samples.
        if n_start < self.n_classes:
            print('n_start', n_start, ' number of points is less than the number of classes', self.n_classes, ', so we change it.')
            n_start = self.n_classes

        # Sample n_start datapoints.
        self.indices_known = []
        self.indices_unknown = []
        for i in np.unique(self.dataset.train_labels):
            # First get 1 point from each class.
            cl = np.nonzero(self.dataset.train_labels==i)[0]
            # Insure that we select random datapoints.
            indices = np.random.permutation(cl)
            self.indices_known.append(indices[0])
            self.indices_unknown.extend(indices[1:])
        self.indices_known = np.array(self.indices_known)
        self.indices_unknown = np.array(self.indices_unknown)
        print("self.indices_known = np.array(self.indices_known)")
        print("self.indices_known length", len(self.indices_known))
        print("\n")
        print("self.indices_unknown length", len(self.indices_unknown))
        print("\n")

        # The self.indices_unknown now contains first all points of class_1, then all points of class_2 etc.
        # So, we permute them.
        self.indices_unknown = np.random.permutation(self.indices_unknown)
        print("self.indices_unknown = np.random.permutation(self.indices_unknown)")
        print("self.indices_unknown length", len(self.indices_unknown))
        print("\n")

        # Then, sample the rest of the datapoints at random.
        if n_start > self.n_classes:
            self.indices_known = np.concatenate(([self.indices_known, self.indices_unknown[0:n_start-self.n_classes]]))
            self.indices_unknown = self.indices_unknown[n_start-self.n_classes:]
            
        # BUILD AN INITIAL MODEL.

        # Get the data corresponding to the selected indices.
        known_data = self.dataset.train_data[self.indices_known,:]
        print("known_data length", len(known_data))
        print("\n")
        known_labels = self.dataset.train_labels[self.indices_known]
        unknown_data = self.dataset.train_data[self.indices_unknown,:]
        print("unknown_data length", len(unknown_data))
        print("\n")
        unknown_labels = self.dataset.train_labels[self.indices_unknown]

        # Train a model using data corresponding to indices_known:
        known_labels = np.ravel(known_labels)
        self.model.fit(known_data, known_labels)

        # Compute the quality score:
        test_prediction = self.model.predict(self.dataset.test_data)
        print("test_prediction = self.model.predict(self.dataset.test_data)")
        print("test_prediction length", len(test_prediction))
        print("\n")
        new_score = self.quality_method(self.dataset.test_labels, test_prediction)
        print("new_score = self.quality_method(self.dataset.test_labels, test_prediction)")
        print(new_score)
        print("\n")
        self.episode_qualities.append(new_score)
        print("self.episode_qualities.append(new_score)")
        print(self.episode_qualities)
        print("\n")

        # Get the features categorizing the state.
        print("GET STATE!") 
        state, next_action = self._get_state()
        print("\n\n\n")
        print("BACK TO ENV RESET")
        print("state length", len(state))
        print("\n")
        print("next_action length", len(next_action))
        print("\n")
        self.n_actions = np.size(self.indices_unknown)
        print("self.n_actions", self.n_actions)
        print("\n")

        return state, next_action
        


    def step(self, batch_actions_indices):

        print("\n\n\n")
        print("STEP ENVS")

        # The batch_actions_indices value indicates the positions
        # of the batch of datapoints in self.indices_unknown that we want to sample in unknown_data.
        # The index in train_data should be retrieved.
        selection_absolute = self.indices_unknown[batch_actions_indices]
        print("selection_absolute = self.indices_unknown[batch_actions_indices]")
        print("length batch_actions_indices", len(batch_actions_indices))
        print("\n")
        print("length selection_absolute", len(selection_absolute))
        print("\n")

        # Label a datapoint: add its index to known samples and remove from unknown.
        self.indices_known = np.concatenate((self.indices_known, selection_absolute))
        print("self.indices_known = np.concatenate((self.indices_known, selection_absolute))  ")
        print("length self.indices_known", len(self.indices_known))
        print("\n")
        self.indices_unknown = np.delete(self.indices_unknown, batch_actions_indices)
        print("self.indices_unknown = np.delete(self.indices_unknown, batch_actions_indices)")
        print("length self.indices_unknown", len(self.indices_unknown))
        print("\n")  

        # Train a model with new labeled data:
        known_data = self.dataset.train_data[self.indices_known,:]
        print("known_data = self.dataset.train_data[self.indices_known,:]")
        print("length known_data", len(known_data))
        print("\n")
        known_labels = self.dataset.train_labels[self.indices_known]
        known_labels = np.ravel(known_labels)
        self.model.fit(known_data, known_labels)

        # Get a new state.
        print("GO TO GET STATE")
        state, next_action = self._get_state()
        print("next_action length", len(next_action))
        print("\n")

        # Update the number of available actions.
        self.n_actions = np.size(self.indices_unknown)
        print("n_actions", self.n_actions)

        # Compute the quality of the current classifier.
        test_prediction = self.model.predict(self.dataset.test_data)
        print("test_prediction = self.model.predict(self.dataset.test_data)")
        print("test_prediction length", len(test_prediction))
        print("\n")
        new_score = self.quality_method(self.dataset.test_labels, test_prediction)
        print("new_score", new_score)
        print("\n")
        self.episode_qualities.append(new_score)
        print("self.episode_qualities.append(new_score)")
        print("self.episode_qualities", self.episode_qualities)
        print("\n")

        # Compute the reward.
        reward = self._compute_reward()
        print("reward = self._compute_reward()")
        print("reward", reward)

        # Check if this episode terminated.
        done = self._compute_is_terminal()
        print("\n\n\n")
        print("done", done)
        print("\n\n\n")

        print("state length", len(state))
        print("next_action length", len(next_action))
        print("RETURN")
        print("\n\n\n")
        return state, next_action, reward, done
      


    def _get_state(self):

        print("\n\n")
        print("INSIDE GET STATE")
        # COMPUTE state.
        predictions = self.model.predict_proba(self.dataset.state_data)[:,0]
        print("predictions = self.model.predict_proba(self.dataset.state_data)[:,0]")
        print("predictions length", len(predictions))
        print("\n")
        predictions = np.array(predictions)
        print("predictions = np.array(predictions)")
        print("predictions length", len(predictions))
        print("\n")
        idx = np.argsort(predictions)
        print("idx = np.argsort(predictions)")
        print("idx", idx)
        print("idx length", len(idx))
        print("\n")

        # The state representation is the *sorted* list of scores.
        state = predictions[idx]
        print("state = predictions[idx]")
        print("state length", len(state))
        print("\n")
        
        # COMPUTE next_action.
        unknown_data = self.dataset.train_data[self.indices_unknown,:]
        print("unknown_data = self.dataset.train_data[self.indices_unknown,:]")
        print("unknown_data length", len(unknown_data))
        print("\n")

        next_action = []
        for i in range(1, len(unknown_data)+1):
            next_action.append(np.array([i]))
        print("next_action length", len(next_action))
        print("\n")
        print("RETURN")
        return state, next_action
    


    def _compute_reward(self):

        reward = 0.0
        
        return reward
    


    def _compute_is_terminal(self):

        # The self.n_actions contains a number of unlabeled datapoints that are left.
        if self.n_actions==0:
            print('We ran out of samples!')
            done = True
        else:
            done = False
   
        return done
        
    
    


class LalEnvFirstAccuracy(LalEnv): 

    def __init__(self, dataset, model, quality_method):

        # Inits environment with its normal attributes.
        LalEnv.__init__(self, dataset, model, quality_method)
    

        
    def reset(self, n_start=2):

        state, next_action = LalEnv.reset(self, n_start=n_start)
        current_reward = self._compute_reward()

        # Store the current rewatd.
        self.rewards_bank.append(current_reward)
        
        return state, next_action, current_reward
       


    def _compute_reward(self):

        # Find the reward as new_score - previous_score.


        new_score = self.episode_qualities[-1]
        previous_score = self.episode_qualities[-2]
        reward = new_score - previous_score
        self.rewards_bank.append(reward)
        return reward
    
    

    def _compute_is_terminal(self):

        print("\n")
        print("COMPUTE IS TERMINAL")

        # By default the episode will terminate when all samples are labelled.
        done = LalEnv._compute_is_terminal(self)
 
        # If the last three rewards are declining, then terminate the episode.
        if len(self.rewards_bank) >= 3:
            print("self.rewards_bank[-1]", self.rewards_bank[-1])
            print("self.rewards_bank[-2]", self.rewards_bank[-2])
            print("self.rewards_bank[-3]", self.rewards_bank[-3])

            if self.rewards_bank[-1] < self.rewards_bank[-2] and self.rewards_bank[-2] < self.rewards_bank[-3]:
                done = True
                print("Rewards ARE DECLINING!")
                print("Done", done)
                print("\n\n\n")
                return done
        print("Rewards are NOOOT declining")
        print("\n\n\n")
        return done
             


    def return_episode_qualities(self):
        return self.episode_qualities