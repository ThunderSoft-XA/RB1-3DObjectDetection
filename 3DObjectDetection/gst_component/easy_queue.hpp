#ifndef Queue_HPP
#define Queue_HPP

#include<iostream>
#include<memory>
#include<queue>
#include<mutex>
#include<condition_variable>

template <typename T>
class Queue 
{
public:
    Queue();
    ~Queue();

    T& front();
    T pop();
    void pop_front();
    void push_back(const T& item);
    void push_back(T&& item);
    void shut_down();

    // bool pop(std::unique_ptr<T>&& pData);
    // void push(std::unique_ptr<T>&& item);
    int size();
    bool empty();
    bool is_shutdown();

public:
    T operator[] (int k) {
        return m_queue[k];
    }
	Queue &operator= (Queue &p) {
		this->m_queue = p.m_queue;
        return *this;
    }
private:
    std::deque<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    bool m_bShutDown = false;
};

// #include"Queue.hpp"
template <typename T>
Queue<T>::Queue(){

}

template <typename T>
Queue<T>::~Queue(){

}

template <typename T>
bool Queue<T>::is_shutdown() {
    return m_bShutDown;
}

template <typename T>
void Queue<T>::shut_down() {
    std::unique_lock<std::mutex> mlock(m_mutex);
    m_queue.clear();
    m_bShutDown = true;
}

template <typename T>
T Queue<T>::pop()
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    // if (m_queue.empty()) {
    //     return false;
    // }

    m_cond.wait(mlock, [this](){return !this->m_queue.empty();});
    T rc(std::move(m_queue.front()));
    m_queue.pop_front();

    return rc;
}

template <typename T>
T& Queue<T>::front()
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    while(m_queue.empty()) {
        m_cond.wait(mlock);
    }    

    return m_queue.front();
}

template <typename T>
void Queue<T>::pop_front()
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    while(m_queue.empty()) {
        m_cond.wait(mlock);
    }

    m_queue.pop_front();
}

template <typename T>
void Queue<T>::push_back(const T& item)
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    m_queue.push_back(item);
    mlock.unlock();
    m_cond.notify_one();
}

template <typename T>
void Queue<T>::push_back(T&& item)
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    m_queue.push_back(std::move(item));
    mlock.unlock();
    m_cond.notify_one();  
}

template <typename T>
int Queue<T>::size()
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    int size = m_queue.size();
    mlock.unlock();
    return size;
}

template <typename T>
bool Queue<T>::empty()
{
    std::unique_lock<std::mutex> mlock(m_mutex);
    bool is_empty = m_queue.empty();
    mlock.unlock();
    return is_empty;
}

#endif