#include <qurt.h>

#include <cstdint>

#include "util.hpp"

namespace hexagon {

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
    }

    ~qurt_thread() {
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

template <size_t _thread_count> class thread_pool {
  public:
    thread_pool();
    ~thread_pool();

  private:
    thread_pool(const thread_pool &)    = delete;
    void operator=(const thread_pool &) = delete;
    thread_pool(thread_pool &&)         = delete;
    void operator=(thread_pool &&)      = delete;
};

}  // namespace hexagon
