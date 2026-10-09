import great_expectations as ge
from great_expectations.core.expectation_configuration import ExpectationConfiguration

def create_expectation_suite(context):
    suite = context.create_expectation_suite("credit_data_suite", overwrite=True)

    expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "LIMIT_BAL"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "LIMIT_BAL"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "AGE", "min_value": 18, "max_value": 100}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "default.payment.next.month", "value_set": [0, 1]}
        ),
    ]

    for exp in expectations:
        suite.add_expectation(exp)

    context.save_expectation_suite(suite, "credit_data_suite")
    return suite