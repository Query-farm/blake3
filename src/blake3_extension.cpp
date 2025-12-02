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

static void LoadInternal(ExtensionLoader &loader) {
	auto agg_set = AggregateFunctionSet("blake3");

	auto agg_varchar =
	    AggregateFunction::UnaryAggregate<Blake3State, string_t, string_t, Blake3OperationWithSize<Blake3State>>(
	        LogicalType::VARCHAR, LogicalType::BLOB);
	agg_varchar.name = "blake3";
	agg_varchar.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
	agg_varchar.distinct_dependent = AggregateDistinctDependent::DISTINCT_DEPENDENT;
	agg_set.AddFunction(agg_varchar);

	auto agg_blob =
	    AggregateFunction::UnaryAggregate<Blake3State, string_t, string_t, Blake3OperationWithSize<Blake3State>>(
	        LogicalType::BLOB, LogicalType::BLOB);
	agg_blob.name = "blake3";
	agg_blob.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
	agg_blob.distinct_dependent = AggregateDistinctDependent::DISTINCT_DEPENDENT;
	agg_set.AddFunction(agg_blob);

	auto agg_bigint = AggregateFunction::UnaryAggregate<Blake3State, int64_t, string_t, Blake3Operation<Blake3State>>(
	    LogicalType::BIGINT, LogicalType::BLOB);
	agg_bigint.name = "blake3";
	agg_bigint.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
	agg_bigint.distinct_dependent = AggregateDistinctDependent::DISTINCT_DEPENDENT;
	agg_set.AddFunction(agg_bigint);

	loader.RegisterFunction(agg_set);

	//	loader.RegisterFunction(agg_with_arg);
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
