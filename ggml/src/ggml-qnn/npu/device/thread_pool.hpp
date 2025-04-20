#include <qurt.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "util.hpp"

namespace hexagon {

constexpr const size_t             kDefaultStackSize       = 1024 * 16;  // 16KB
constexpr const unsigned long long kThreadTaskPendingBit   = 1;
constexpr const unsigned long long kThreadTaskCompletedBit = 2;

template <size_t _stack_size> class qurt_thread {
  public:
    typedef void (*qurt_thread_func_type)(qurt_thread * thread, void * arg);

    qurt_thread(const char * thread_name, qurt_thread_func_type thread_func, void * arg, unsigned short priority) {
        qurt_thread_attr_init(&_attributes);
        qurt_thread_attr_set_name(&_attributes, (char *) thread_name);
        qurt_thread_attr_set_stack_addr(&_attributes, _stack);
        qurt_thread_attr_set_stack_size(&_attributes, _stack_size);
        qurt_thread_attr_set_priority(&_attributes, priority);

        auto ret = qurt_thread_create(&_tid, &_attributes, qurt_thread::thread_func_impl, (void *) &this);
        if (ret != QURT_EOK) {
            DEVICE_LOG_ERROR("Failed to create thread: %d", (int) ret);
            return;
        }

        _func = thread_func;
        _arg  = arg;
        DEVICE_LOG_DEBUG("qurt_thread.created: %s, id: %d", thread_name, (int) _tid);
    }

    ~qurt_thread() {
        DEVICE_LOG_DEBUG("qurt_thread.destroy: %d", (int) _tid);
        int  thread_exit_code = QURT_EOK;
        auto ret              = qurt_thread_join(_tid, &thread_exit_code);
        if (ret != QURT_EOK || ret != QURT_ENOTHREAD) {
            DEVICE_LOG_ERROR("Failed to join thread: %d", (int) ret);
            return;
        }

        if (thread_exit_code != QURT_EOK) {
            DEVICE_LOG_ERROR("Thread exit code: %d", (int) thread_exit_code);
        }
    }

    bool is_valid() const { return _tid != 0 && _func != nullptr; }

  private:
    static void thread_func_impl(qurt_thread * thread) {
        if (thread->_func) {
            thread->_func(thread, thread->_arg);
        }

        qurt_thread_exit(QURT_EOK);
    }

    uint8_t               _stack[_stack_size];
    qurt_thread_t         _tid;
    qurt_thread_attr_t    _attributes;
    qurt_thread_func_type _func = nullptr;
    void *                _arg  = nullptr;

    qurt_thread(const qurt_thread &)    = delete;
    void operator=(const qurt_thread &) = delete;
    qurt_thread(qurt_thread &&)         = delete;
    void operator=(qurt_thread &&)      = delete;
};

using quart_thread_ptr = std::unique_ptr<qurt_thread<kDefaultStackSize>>;

template <size_t _thread_count> class thread_pool {
  public:
    typedef void (*task_type)(thread_pool * pool, size_t thread_idx, size_t thread_count, void * arg);

    thread_pool() {
        std::string thread_name = "thread_pool_";
        for (size_t i = 0; i < _thread_count; ++i) {
            _thread_args[i].pool       = this;
            _thread_args[i].thread_idx = i;
            qurt_signal_init(&_thread_args[i].signal);

            auto thread = std::make_unique<qurt_thread<kDefaultStackSize>>(
                thread_name + std::to_string(i), thread_pool::thread_func_impl, this, QURT_THREAD_PRIORITY_DEFAULT);
            if (!thread->is_valid()) {
                DEVICE_LOG_ERROR("Failed to create thread: %zu", i);
                return;
            }

            _threads[i] = std::move(thread);
        }
        DEVICE_LOG_DEBUG("thread_pool.created: %zu", _thread_count);
    }

    ~thread_pool() {
        DEVICE_LOG_DEBUG("thread_pool.destroy");
        _thread_exit = true;
        for (auto & thread : _threads) {
            thread.reset();
        }

        for (auto & arg : _thread_args) {
            if (arg.pool) {
                qurt_signal_destroy(&arg.signal);
            }
        }
    }

    bool sync_execute(task_type task, void * arg) {
        if (!task) {
            DEVICE_LOG_ERROR("Invalid task");
            return false;
        }

        _task = task;
        _arg  = arg;
        for (size_t i = 0; i < _thread_count; ++i) {
            qurt_signal_set(&_thread_args[i].signal, kThreadTaskPendingBit);
        }

        for (size_t i = 0; i < _thread_count; ++i) {
            while (
                !(qurt_signal_wait_all(&_thread_args[i].signal, kThreadTaskCompletedBit) & kThreadTaskCompletedBit)) {
                // spurious wakeup? should we clear the signal?
                DEVICE_LOG_DEBUG("thread_pool.sync_execute.spurious_wakeup: %zu", i);
            }
            qurt_signal_clear(&_thread_args[i].signal, kThreadTaskCompletedBit);
        }

        return true;
    }

  private:
    struct thread_pool_arg {
        thread_pool * pool;
        size_t        thread_idx;
        qurt_signal_t signal;
    };

    static void thread_func_impl(qurt_thread * thread, thread_pool_arg * arg) {
        DEVICE_LOG_DEBUG("thread_func_impl.start: %zu", arg->thread_idx);

        while (!arg->pool->_thread_exit) {
            if (!(qurt_signal_wait_all(&arg->signal, kThreadTaskPendingBit) & kThreadTaskPendingBit)) {
                // spurious wakeup? should we clear the signal?
                DEVICE_LOG_DEBUG("thread_func_impl.spurious_wakeup: %zu", arg->thread_idx);
                continue;
            }

            qurt_signal_clear(&arg->signal, kThreadTaskPendingBit);
            if (arg->pool->_thread_exit) {
                DEVICE_LOG_DEBUG("thread_func_impl.exit: %zu", arg->thread_idx);
                break;
            }

            auto task = arg->pool->_thread_args[arg->thread_idx].task;
            if (task) {
                task(arg->pool, arg->thread_idx, arg->pool->_thread_count,
                     arg->pool->_thread_args[arg->thread_idx].arg);
            }

            DEVICE_LOG_DEBUG("thread_func_impl.task_completed: %zu", arg->thread_idx);
            qurt_signal_set(&arg->signal, kThreadTaskCompletedBit);
        }

        DEVICE_LOG_DEBUG("thread_func_impl.end: %zu", arg->thread_idx);
    }

    std::atomic_bool                            _thread_exit = false;
    std::array<quart_thread_ptr, _thread_count> _threads;
    thread_pool_arg                             _thread_args[_thread_count] = {};
    task_type                                   _task                       = nullptr;
    void *                                      _arg                        = nullptr;

    thread_pool(const thread_pool &)    = delete;
    void operator=(const thread_pool &) = delete;
    thread_pool(thread_pool &&)         = delete;
    void operator=(thread_pool &&)      = delete;
};

}  // namespace hexagon
