//
// Created by Jonas on 29/03/2022, based on a similar template by Wim Vancroonenburg
//

#ifndef MESHCORE_LOCALSEARCH_H
#define MESHCORE_LOCALSEARCH_H

#include <chrono>
#include <memory>
#include <type_traits>
#include <functional>
#include "../utility/random.h"
#include "AbstractSolution.h"

template <class Solution>
class ObjectiveFunction {
private:
    static_assert(std::is_base_of<AbstractSolution, Solution>::value, "ObjectiveFunction template type should be a solution derived from AbstractSolution class");
public:
    virtual ~ObjectiveFunction() {};
    [[nodiscard]] virtual double evaluate(std::shared_ptr<const Solution> s) const = 0;
};

template <class Solution>
class Move {
private:
    static_assert(std::is_base_of<AbstractSolution, Solution>::value, "Move template type should be a solution derived from AbstractSolution class");
public:
	virtual ~Move(){}
	virtual void doMove(std::shared_ptr<Solution> solution) = 0;
	virtual void undoMove(std::shared_ptr<Solution> solution) = 0;
};

template <class S>
class MoveFactory {
private:
    static_assert(std::is_base_of<AbstractSolution, S>::value, "MoveFactory template type should be a solution derived from AbstractSolution class");
public:
	virtual ~MoveFactory(){};
    [[nodiscard]] virtual std::shared_ptr<Move<S>> sample(std::shared_ptr<const S> s, const Random& random, double stepSize) const = 0;
    virtual std::vector<std::shared_ptr<Move<S>>> listMoves(std::shared_ptr<const S> s, double stepSize) const {
        return {};
    };
};

template <class S>
class AbstractLocalSearchListener {
private:
    static_assert(std::is_base_of<AbstractSolution, S>::value, "AbstractLocalSearchListener template type should be a solution derived from AbstractSolution class");

public:
    virtual void started(std::shared_ptr<const S> initialSolution, double initialScore) const = 0;
    virtual void finished(std::shared_ptr<const S> bestSolution, double bestScore) const = 0;
    virtual void foundNewBestSolution(std::shared_ptr<const S> bestSolution, double bestScore, unsigned int currentIteration) const = 0;
    virtual void foundNewCurrentSolution(std::shared_ptr<const S> currentSolution, double currentScore, std::shared_ptr<const S> bestSolution, double bestScore, unsigned int currentIteration) const = 0;
};

template<class S>
class LocalSearchListener: public AbstractLocalSearchListener<S> {
private:
    static_assert(std::is_base_of<AbstractSolution, S>::value, "LocalSearchListener template type should be a solution derived from AbstractSolution class");
    std::function<void(std::shared_ptr<const S> initialSolution, double initialScore)> onStarted = {};
    std::function<void(std::shared_ptr<const S> bestSolution, double bestScore)> onFinished = {};
    std::function<void(std::shared_ptr<const S> bestSolution, double bestScore, unsigned int currentIteration)> onNewBestSolutionFound = {};
    std::function<void(std::shared_ptr<const S> currentSolution, double currentScore, std::shared_ptr<const S> bestSolution, double bestScore, unsigned int currentIteration)> onNewCurrentSolutionFound = {};

    void started(std::shared_ptr<const S> initialSolution, double initialScore) const override {
        if(this->onStarted) this->onStarted(initialSolution, initialScore);
    }

    void finished(std::shared_ptr<const S> bestSolution, double bestScore) const override {
        if(this->onFinished) this->onFinished(bestSolution, bestScore);
    }

    void foundNewBestSolution(std::shared_ptr<const S> bestSolution, double bestScore, unsigned int currentIteration) const override {
        if(this->onNewBestSolutionFound) this->onNewBestSolutionFound(bestSolution, bestScore, currentIteration);
    }

    void foundNewCurrentSolution(std::shared_ptr<const S> currentSolution, double currentScore, std::shared_ptr<const S> bestSolution, double bestScore, unsigned int currentIteration) const override {
        if(this->onNewCurrentSolutionFound) this->onNewCurrentSolutionFound(currentSolution, currentScore, bestSolution, bestScore, currentIteration);
    }

public:
    [[maybe_unused]] void setOnStarted(const std::function<void(std::shared_ptr<const S>, double)> &newOnStarted) {
        this->onStarted = newOnStarted;
    }

    [[maybe_unused]] void setOnFinished(const std::function<void(std::shared_ptr<const S>, double)> &newOnFinished) {
        this->onFinished = newOnFinished;
    }

    [[maybe_unused]] void setOnNewBestSolutionFound(const std::function<void(std::shared_ptr<const S>, double, unsigned int)> &newOnNewBestSolutionFound) {
        this->onNewBestSolutionFound = newOnNewBestSolutionFound;
    }

    [[maybe_unused]] void setOnNewCurrentSolutionFound(const std::function<void(std::shared_ptr<const S>, double, std::shared_ptr<const S>, double, unsigned int)> &newOnNewCurrentSolutionFound) {
        this->onNewCurrentSolutionFound = newOnNewCurrentSolutionFound;
    }
};

template <class S>
class AbstractLocalSearch {
private:
//    static_assert(std::is_abstract<S>::value || std::is_default_constructible<S>::value, "AbstractLocalSearch template type should be default constructable");
    static_assert(std::is_base_of<AbstractSolution, S>::value, "AbstractLocalSearch template type should be a solution derived from AbstractSolution class");
private:
    std::vector<std::shared_ptr<AbstractLocalSearchListener<S>>> listeners;
protected:
    bool stopped = false;
    const unsigned int maximumIterations;
    const ObjectiveFunction<S>& objectiveFunction;
    const MoveFactory<S>& moveFactory;
    const Random& random;

public:
    AbstractLocalSearch(MoveFactory<S>& moveFactory, const ObjectiveFunction<S>& objectiveFunction, unsigned int maximumIterations, const Random& random):
    maximumIterations(maximumIterations),
    objectiveFunction(objectiveFunction),
    moveFactory(moveFactory),
    random(random){
    }

    std::shared_ptr<S> executeSearch(std::shared_ptr<S> initialSolution){

        // Initialize algorithm specific data structures
        this->stopped = false;
        auto initialScore = objectiveFunction.evaluate(initialSolution);
        this->initialize(initialSolution, initialScore);

        // Initialise current and best results
        std::shared_ptr<S> currentSolution = std::static_pointer_cast<S>(initialSolution->clone());
        double currentScore = initialScore;
        std::shared_ptr<S> bestSolution = std::static_pointer_cast<S>(currentSolution->clone());
        double bestScore = currentScore;

        // Notify listeners and start optimization loop
        this->notifyListenersStarted(currentSolution, currentScore);
        for (int currentIteration=0; currentIteration < maximumIterations; ++currentIteration){

            if(stopped) break;

            currentScore = performIteration(currentIteration, currentSolution, currentScore);
            assert(currentScore == objectiveFunction.evaluate(currentSolution));
            this->notifyListenersFoundNewCurrentSolution(currentSolution, currentScore, bestSolution, bestScore, currentIteration);

            // Check if we improved on the best solution
            if(currentScore<bestScore){
                bestScore = currentScore;
                bestSolution = std::static_pointer_cast<S>(currentSolution->clone());
                this->notifyListenersFoundNewBestSolution(bestSolution, bestScore, currentIteration);
            }
        }

        this->notifyListenersFinished(bestSolution, bestScore);
        return bestSolution;
    }

    virtual ~AbstractLocalSearch(){};
    void addListener(std::shared_ptr<AbstractLocalSearchListener<S>> listener){
        this->listeners.emplace_back(listener);
    }
private:
    virtual double performIteration(unsigned int currentIteration, std::shared_ptr<S> currentSolution, double currentScore) = 0;
    virtual void initialize(std::shared_ptr<S> initialSolution, double initialScore) = 0;

protected:
    void notifyListenersFinished(std::shared_ptr<const S> solution, double score){
        for(const auto& listener: listeners){
            listener->finished(solution, score);
        }
    }
    void notifyListenersStarted(std::shared_ptr<const S> initialSolution, double initialScore){
        for(const auto& listener: listeners){
            listener->started(initialSolution, initialScore);
        }
    }
    void notifyListenersFoundNewBestSolution(std::shared_ptr<const S> bestSolution, double bestScore, unsigned int iteration){
        for(const auto& listener: listeners){
            listener->foundNewBestSolution(bestSolution, bestScore, iteration);
        }
    }
    void notifyListenersFoundNewCurrentSolution(std::shared_ptr<const S> currentSolution, double currentScore, std::shared_ptr<const S> bestSolution, double bestScore, unsigned int currentIteration){
        for(const auto& listener: this->listeners){
            listener->foundNewCurrentSolution(currentSolution, currentScore, bestSolution, bestScore, currentIteration);
        }
    }
};

template <class S>
class CompositeMoveFactory: public MoveFactory<S> {
private:
    std::vector<std::shared_ptr<MoveFactory<S>>> moveFactories;
    std::vector<double> probabilities; // The total sum should be 1.0

public:
	explicit CompositeMoveFactory(const std::vector<std::shared_ptr<MoveFactory<S>>>& mfs, std::vector<double> weights = {}) : moveFactories(mfs) {
        if(!weights.empty()){
            double totalWeight = 0;
            for(auto& weight : weights){
                totalWeight += weight;
            }
            for(auto& weight : weights){
                probabilities.emplace_back(weight/totalWeight);
            }
        }
	}
	virtual ~CompositeMoveFactory(){}

//    [[nodiscard]] virtual std::shared_ptr<Move<S>> sample(std::shared_ptr<const S> s, const Random& random) const = 0;
    std::shared_ptr<Move<S>> sample(std::shared_ptr<const S> s, const Random& random, double stepSize) const override {
		int idx;
		if(probabilities.empty()){
            idx = random.nextInteger(0, moveFactories.size()-1);
		}
		else{
			idx = rouletteWheel(probabilities, random);
		}
		return moveFactories.at(idx)->sample(s, random, stepSize);
	}

    std::vector<std::shared_ptr<Move<S>>>
    listMoves(std::shared_ptr<const S> s, double stepSize) const override {
        std::vector<std::shared_ptr<Move<S>>> moves;
        for(const auto& moveFactory: moveFactories) {
            const auto factoryMoves = moveFactory->listMoves(s, stepSize);
            moves.insert(moves.end(), factoryMoves.begin(), factoryMoves.end());
        }
        return moves;
    }

private:
    /** Returns a random index with given probabilities **/
    static int rouletteWheel(const std::vector<double>& probabilities, const Random& random){
        double rnd = random.nextDouble();
        for(int i = 0; i<probabilities.size(); i++) {
            const auto& probability = probabilities.at(i);
            if(probability >= rnd){
                return i;
            }
            else rnd -= probability; // We know this is still > 0.0, because probability < rnd
        }
        return -1;
    }
};

#endif //MESHCORE_LOCALSEARCH_H