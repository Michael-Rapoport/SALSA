if __name__ == "__main__":
    # Initialize and train meta-learner
    meta_learner = MetaLearner(train_data, test_data, num_workers=2, meta_epochs=5)
    meta_learner.train()