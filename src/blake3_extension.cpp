#define DUCKDB_EXTENSION_MAIN

#include "blake3_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "blake3.h"

namespace duckdb {

struct Blake3State {
	blake3_hasher hasher;
	bool did_update = false;

	Blake3State() {
		blake3_hasher_init(&hasher);
	}

	~Blake3State() {
		blake3_hasher_reset(&hasher);
	}
};

template <class STATE_DATA_TYPE>
struct Blake3OperationWithSize {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.did_update = false;
		blake3_hasher_init(&state.hasher);
	}

	static bool IgnoreNull() {
		return true;
	}

	template <class A_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const A_TYPE &a_data, AggregateUnaryInput &idata) {
		if (!state.did_update) {
			state.did_update = true;
		}
		uint64_t size = a_data.GetSize();
		// hash the record length as well to prevent length extension attacks
		blake3_hasher_update(&state.hasher, &size, sizeof(uint64_t));
		blake3_hasher_update(&state.hasher, a_data.GetDataUnsafe(), size);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input,
	                              idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input_data) {
		if (source.did_update && !target.did_update) {
			target.hasher = source.hasher;
			target.did_update = true;
		} else if (!source.did_update) {
			// nothing to do
			return;
		}
		throw InvalidInputException("Blake3 hash requires a distinct total ordering, example: blake3(data ORDER BY "
		                            "data) columns in group by clauses are not sufficient");
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.did_update) {
			finalize_data.ReturnNull();
		} else {
			char output[BLAKE3_OUT_LEN];
			blake3_hasher_finalize(&state.hasher, reinterpret_cast<uint8_t *>(&output), BLAKE3_OUT_LEN);
			target = StringVector::AddStringOrBlob(finalize_data.result, reinterpret_cast<const char *>(&output),
			                                       BLAKE3_OUT_LEN);
		}
	}
};

template <class STATE_DATA_TYPE>
struct Blake3Operation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.did_update = false;
		blake3_hasher_init(&state.hasher);
	}

	static bool IgnoreNull() {
		return true;
	}

	template <class A_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const A_TYPE &a_data, AggregateUnaryInput &idata) {
		if (!state.did_update) {
			state.did_update = true;
		}
		blake3_hasher_update(&state.hasher, &a_data, sizeof(A_TYPE));
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input,
	                              idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &aggr_input_data) {
		if (source.did_update && !target.did_update) {
			target.hasher = source.hasher;
			target.did_update = true;
		} else if (!source.did_update) {
			// nothing to do
			return;
		}
		throw InvalidInputException("Blake3 hash requires a distinct total ordering, example: blake3(data ORDER BY "
		                            "data) columns in group by clauses are not sufficient");
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.did_update) {
			finalize_data.ReturnNull();
		} else {
			char output[BLAKE3_OUT_LEN];
			blake3_hasher_finalize(&state.hasher, reinterpret_cast<uint8_t *>(&output), BLAKE3_OUT_LEN);
			target = StringVector::AddStringOrBlob(finalize_data.result, reinterpret_cast<const char *>(&output),
			                                       BLAKE3_OUT_LEN);
		}
	}
};

// Scalar function for variable-size types (VARCHAR, BLOB)
static void Blake3ScalarFunctionWithSize(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t input) {
		blake3_hasher hasher;
		blake3_hasher_init(&hasher);

		uint64_t size = input.GetSize();
		blake3_hasher_update(&hasher, &size, sizeof(uint64_t));
		blake3_hasher_update(&hasher, input.GetDataUnsafe(), size);

		char output[BLAKE3_OUT_LEN];
		blake3_hasher_finalize(&hasher, reinterpret_cast<uint8_t *>(&output), BLAKE3_OUT_LEN);

		return StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(&output), BLAKE3_OUT_LEN);
	});
}

// Scalar function for fixed-size types
template <typename CPP_TYPE>
static void Blake3ScalarFunctionFixedSize(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::Execute<CPP_TYPE, string_t>(args.data[0], result, args.size(), [&](CPP_TYPE input) {
		blake3_hasher hasher;
		blake3_hasher_init(&hasher);
		blake3_hasher_update(&hasher, &input, sizeof(CPP_TYPE));

		char output[BLAKE3_OUT_LEN];
		blake3_hasher_finalize(&hasher, reinterpret_cast<uint8_t *>(&output), BLAKE3_OUT_LEN);

		return StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(&output), BLAKE3_OUT_LEN);
	});
}

// Helper function to register a fixed-size type aggregate
template <typename CPP_TYPE>
static void RegisterFixedSizeType(AggregateFunctionSet &agg_set, const LogicalType &logical_type) {
	auto agg_func = AggregateFunction::UnaryAggregate<Blake3State, CPP_TYPE, string_t, Blake3Operation<Blake3State>>(
	    logical_type, LogicalType::BLOB);
	agg_func.name = "blake3";
	agg_func.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
	agg_func.distinct_dependent = AggregateDistinctDependent::DISTINCT_DEPENDENT;
	agg_set.AddFunction(agg_func);
}

// Helper function to register a fixed-size type scalar
template <typename CPP_TYPE>
static void RegisterFixedSizeScalar(ScalarFunctionSet &scalar_set, const LogicalType &logical_type) {
	ScalarFunction scalar_func({logical_type}, LogicalType::BLOB, Blake3ScalarFunctionFixedSize<CPP_TYPE>);
	scalar_func.name = "blake3_hash";
	scalar_set.AddFunction(scalar_func);
}

// Helper function to register a variable-size type aggregate (VARCHAR, BLOB)
template <typename CPP_TYPE>
static void RegisterVariableSizeType(AggregateFunctionSet &agg_set, const LogicalType &logical_type) {
	auto agg_func =
	    AggregateFunction::UnaryAggregate<Blake3State, CPP_TYPE, string_t, Blake3OperationWithSize<Blake3State>>(
	        logical_type, LogicalType::BLOB);
	agg_func.name = "blake3";
	agg_func.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
	agg_func.distinct_dependent = AggregateDistinctDependent::DISTINCT_DEPENDENT;
	agg_set.AddFunction(agg_func);
}

// Helper function to register a variable-size type scalar (VARCHAR, BLOB)
static void RegisterVariableSizeScalar(ScalarFunctionSet &scalar_set, const LogicalType &logical_type) {
	ScalarFunction scalar_func({logical_type}, LogicalType::BLOB, Blake3ScalarFunctionWithSize);
	scalar_func.name = "blake3_hash";
	scalar_set.AddFunction(scalar_func);
}

static void LoadInternal(ExtensionLoader &loader) {
	// Register aggregate functions
	auto agg_set = AggregateFunctionSet("blake3");

	// Variable-size types (include size prefix to prevent length extension attacks)
	RegisterVariableSizeType<string_t>(agg_set, LogicalType::VARCHAR);
	RegisterVariableSizeType<string_t>(agg_set, LogicalType::BLOB);

	// Fixed-size integer types
	RegisterFixedSizeType<int8_t>(agg_set, LogicalType::TINYINT);
	RegisterFixedSizeType<int16_t>(agg_set, LogicalType::SMALLINT);
	RegisterFixedSizeType<int32_t>(agg_set, LogicalType::INTEGER);
	RegisterFixedSizeType<int64_t>(agg_set, LogicalType::BIGINT);
	RegisterFixedSizeType<uint8_t>(agg_set, LogicalType::UTINYINT);
	RegisterFixedSizeType<uint16_t>(agg_set, LogicalType::USMALLINT);
	RegisterFixedSizeType<uint32_t>(agg_set, LogicalType::UINTEGER);
	RegisterFixedSizeType<uint64_t>(agg_set, LogicalType::UBIGINT);
	RegisterFixedSizeType<hugeint_t>(agg_set, LogicalType::HUGEINT);
	RegisterFixedSizeType<uhugeint_t>(agg_set, LogicalType::UHUGEINT);

	// Fixed-size floating point types
	RegisterFixedSizeType<float>(agg_set, LogicalType::FLOAT);
	RegisterFixedSizeType<double>(agg_set, LogicalType::DOUBLE);

	loader.RegisterFunction(agg_set);

	// Register scalar functions
	auto scalar_set = ScalarFunctionSet("blake3_hash");

	// Variable-size types (include size prefix to prevent length extension attacks)
	RegisterVariableSizeScalar(scalar_set, LogicalType::VARCHAR);
	RegisterVariableSizeScalar(scalar_set, LogicalType::BLOB);

	// Fixed-size integer types
	RegisterFixedSizeScalar<int8_t>(scalar_set, LogicalType::TINYINT);
	RegisterFixedSizeScalar<int16_t>(scalar_set, LogicalType::SMALLINT);
	RegisterFixedSizeScalar<int32_t>(scalar_set, LogicalType::INTEGER);
	RegisterFixedSizeScalar<int64_t>(scalar_set, LogicalType::BIGINT);
	RegisterFixedSizeScalar<uint8_t>(scalar_set, LogicalType::UTINYINT);
	RegisterFixedSizeScalar<uint16_t>(scalar_set, LogicalType::USMALLINT);
	RegisterFixedSizeScalar<uint32_t>(scalar_set, LogicalType::UINTEGER);
	RegisterFixedSizeScalar<uint64_t>(scalar_set, LogicalType::UBIGINT);
	RegisterFixedSizeScalar<hugeint_t>(scalar_set, LogicalType::HUGEINT);
	RegisterFixedSizeScalar<uhugeint_t>(scalar_set, LogicalType::UHUGEINT);

	// Fixed-size floating point types
	RegisterFixedSizeScalar<float>(scalar_set, LogicalType::FLOAT);
	RegisterFixedSizeScalar<double>(scalar_set, LogicalType::DOUBLE);

	loader.RegisterFunction(scalar_set);
}

void Blake3Extension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string Blake3Extension::Name() {
	return "blake3";
}

std::string Blake3Extension::Version() const {
	return "";
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(blake3, loader) {
	duckdb::LoadInternal(loader);
}
}
