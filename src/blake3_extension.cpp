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
struct Blake3Operation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.did_update = false;
		blake3_hasher_init(&state.hasher);
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &aggr_input_data) {
		state.did_update = false;
		blake3_hasher_reset(&state.hasher);
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
		throw InvalidInputException("Blake3 hash requires an ordering, example: blake3_hash(data order by data");
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
	auto agg_set = AggregateFunctionSet("blake3_hash");
	auto agg_with_arg =
	    AggregateFunction::UnaryAggregateDestructor<Blake3State, string_t, string_t, Blake3Operation<Blake3State>,
	                                                AggregateDestructorType::LEGACY>(LogicalType::VARCHAR,
	                                                                                 LogicalType::BLOB);
	agg_with_arg.name = "blake3_hash";
	agg_with_arg.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
	agg_with_arg.distinct_dependent = AggregateDistinctDependent::DISTINCT_DEPENDENT;

	agg_set.AddFunction(agg_with_arg);

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
