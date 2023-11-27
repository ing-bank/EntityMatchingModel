import logging
import re
import time

from emm.loggers.timer import Timer


def test_logging_timer(caplog):
    # enable debug level capture
    caplog.set_level(logging.DEBUG)
    # disable spark from interfering
    logging.getLogger("py4j").setLevel(logging.ERROR)

    with Timer("hello"):
        pass

    assert len(caplog.record_tuples) == 3
    assert caplog.record_tuples[0] == ("emm.loggers.timer", logging.DEBUG, "+> Starting task 'hello'")
    assert caplog.record_tuples[1] == ("emm.loggers.timer", logging.INFO, "hello time: 0.000s")
    assert caplog.record_tuples[2] == ("emm.loggers.timer", logging.DEBUG, "-> Finished task 'hello' in: 0.000s")


def test_logging_timer_stages(caplog):
    # enable debug level capture
    caplog.set_level(logging.DEBUG)
    # disable spark from interfering
    logging.getLogger("py4j").setLevel(logging.ERROR)

    with Timer("hello") as timer:
        timer.label("hello")
        time.sleep(0.3)

        timer.label("world")
        time.sleep(0.7)

        timer.log_params({"msg": "hello world", "n": 3 + 1})

    assert len(caplog.messages) == 6
    assert caplog.messages[0] == "+> Starting task 'hello'"
    assert caplog.messages[1] == "Task 'hello' label 'hello'"
    assert caplog.messages[2] == "Task 'hello' label 'world'"
    assert caplog.messages[3] == "msg=hello world, n=4"
    assert re.match(
        r"hello \(msg=hello world, n=4\) time: 1\.[0-9]{3}s \(setup: 0\.[0-9]{3}s, hello: 0\.3[0-9]{2}s, world: 0\.7[0-9]{2}s\)",
        caplog.messages[4],
    )
    assert caplog.messages[5].startswith("-> Finished task 'hello' in: 1.")
    assert caplog.messages[5].endswith("s")

    assert caplog.record_tuples[0][1] == logging.DEBUG
    assert caplog.record_tuples[1][1] == logging.DEBUG
    assert caplog.record_tuples[2][1] == logging.DEBUG
    assert caplog.record_tuples[3][1] == logging.DEBUG
    assert caplog.record_tuples[4][1] == logging.INFO
    assert caplog.record_tuples[5][1] == logging.DEBUG
