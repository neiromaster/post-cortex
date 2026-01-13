// Copyright (c) 2025 Julius ML
// MIT License

//! Type coercion and validation for MCP tool parameters.
//!
//! This module provides flexible type coercion to help AI agents interact
//! with MCP tools correctly, even when they pass parameters with slightly
//! incorrect types (e.g., numbers instead of strings).

use serde::Deserialize;
use serde_json::Value;

/// Coerce and validate a JSON value into the target type.
///
/// This function attempts to deserialize the value directly first (fast path).
/// If that fails, it applies type coercion rules to fix common type mismatches
/// before attempting deserialization again.
///
/// # Type Coercion Rules
///
/// - Number → String: Converts integers and floats to their string representation
/// - Boolean → String: Converts "true"/"false" to strings
/// - Object/Array → JSON String: Serializes nested structures as JSON strings
///   (useful for `content` fields that expect stringified JSON)
///
/// # Example
///
/// ```rust
/// use serde_json::json;
///
/// #[derive(Deserialize)]
/// struct Request {
///     session_id: String,
///     count: String,
/// }
///
/// let value = json!({
///     "session_id": 123,  // Number instead of string
///     "count": 42         // Number instead of string
/// });
///
/// let req: Request = coerce_and_validate(value)?;
/// assert_eq!(req.session_id, "123");
/// assert_eq!(req.count, "42");
/// ```
pub fn coerce_and_validate<T: for<'de> Deserialize<'de>>(
    value: Value,
) -> Result<T, CoercionError> {
    // Fast path: try direct deserialization first
    if let Ok(result) = serde_json::from_value::<T>(value.clone()) {
        return Ok(result);
    }

    // Slow path: apply coercions and try again
    let coerced = apply_coercions(value)?;
    serde_json::from_value(coerced).map_err(|e| {
        CoercionError::new(
            "Failed to deserialize parameter(s)",
            e,
            None,
        )
    })
}

/// Apply type coercion rules to a JSON value.
///
/// This recursively walks through the value and applies coercion rules
/// to convert common type mismatches.
fn apply_coercions(mut value: Value) -> Result<Value, CoercionError> {
    if let Some(obj) = value.as_object_mut() {
        for (_key, val) in obj.iter_mut() {
            *val = coerce_value(val)?;
        }
    }
    Ok(value)
}

/// Coerce a single value according to the coercion rules.
fn coerce_value(val: &Value) -> Result<Value, CoercionError> {
    match val {
        // Number → String
        // Handles cases like: {"session_id": 123} → {"session_id": "123"}
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::String(i.to_string()))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::String(f.to_string()))
            } else {
                Ok(val.clone())
            }
        }

        // Boolean → String
        // Handles cases like: {"enabled": true} → {"enabled": "true"}
        Value::Bool(b) => Ok(Value::String(b.to_string())),

        // Object → JSON String
        // Handles cases where nested objects need to be stringified
        // e.g., {"metadata": {"key": "value"}} → {"metadata": "{\"key\":\"value\"}"}
        Value::Object(obj) => {
            serde_json::to_string(obj)
                .map(Value::String)
                .map_err(|e| CoercionError::new(
                    "Failed to serialize object to JSON string",
                    e,
                    Some(val.clone()),
                ))
        }

        // Array → JSON String
        // Similar to object coercion, stringifies arrays
        Value::Array(arr) => {
            serde_json::to_string(arr)
                .map(Value::String)
                .map_err(|e| CoercionError::new(
                    "Failed to serialize array to JSON string",
                    e,
                    Some(val.clone()),
                ))
        }

        // String, Null, etc. pass through unchanged
        _ => Ok(val.clone()),
    }
}

/// Structured error type for coercion failures.
///
/// Provides rich error information to help AI agents understand
/// what went wrong and how to fix it.
#[derive(Debug, Clone)]
pub struct CoercionError {
    /// Human-readable error message
    pub message: String,
    /// Path to the parameter that failed (e.g., "session_id", "updates[0].interaction_type")
    pub parameter_path: Option<String>,
    /// Expected type description (e.g., "UUID string", "one of: qa, decision_made, ...")
    pub expected_type: Option<String>,
    /// The actual value that was received
    pub received_value: Option<Value>,
    /// Actionable hint for fixing the error
    pub hint: Option<String>,
}

impl CoercionError {
    /// Create a new coercion error.
    pub fn new(
        message: &str,
        source_error: impl std::error::Error,
        received_value: Option<Value>,
    ) -> Self {
        Self {
            message: format!("{}: {}", message, source_error),
            parameter_path: None,
            expected_type: None,
            received_value,
            hint: None,
        }
    }

    /// Set the parameter path for this error.
    pub fn with_parameter_path(mut self, path: String) -> Self {
        self.parameter_path = Some(path);
        self
    }

    /// Set the expected type description for this error.
    pub fn with_expected_type(mut self, type_desc: &str) -> Self {
        self.expected_type = Some(type_desc.to_string());
        self
    }

    /// Set a hint for fixing this error.
    pub fn with_hint(mut self, hint: &str) -> Self {
        self.hint = Some(hint.to_string());
        self
    }

    /// Convert this error to an MCP error response.
    ///
    /// Creates a structured JSON error that agents can parse and understand.
    pub fn to_mcp_error(&self) -> rmcp::model::ErrorData {
        let mut details = serde_json::json!({
            "message": self.message,
        });

        if let Some(path) = &self.parameter_path {
            details["parameter"] = serde_json::json!(path);
        }

        if let Some(expected) = &self.expected_type {
            details["expectedType"] = serde_json::json!(expected);
        }

        if let Some(received) = &self.received_value {
            details["receivedValue"] = received.clone();
        }

        if let Some(hint) = &self.hint {
            details["hint"] = serde_json::json!(hint);
        }

        rmcp::model::ErrorData::invalid_params(
            serde_json::to_string(&details).unwrap_or_default(),
            None,
        )
    }
}

impl std::fmt::Display for CoercionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(path) = &self.parameter_path {
            write!(f, " (parameter: {})", path)?;
        }
        if let Some(hint) = &self.hint {
            write!(f, "\nHint: {}", hint)?;
        }
        Ok(())
    }
}

impl std::error::Error for CoercionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn test_coerce_number_to_string() {
        let value = json!({"session_id": 123});
        let result: HashMap<String, String> = coerce_and_validate(value).unwrap();
        assert_eq!(result.get("session_id"), Some(&"123".to_string()));
    }

    #[test]
    fn test_coerce_float_to_string() {
        let value = json!({"score": 3.14});
        let result: HashMap<String, String> = coerce_and_validate(value).unwrap();
        assert_eq!(result.get("score"), Some(&"3.14".to_string()));
    }

    #[test]
    fn test_coerce_bool_to_string() {
        let value = json!({"enabled": true});
        let result: HashMap<String, String> = coerce_and_validate(value).unwrap();
        assert_eq!(result.get("enabled"), Some(&"true".to_string()));
    }

    #[test]
    fn test_coerce_object_to_json_string() {
        let value = json!({"metadata": {"key": "value"}});
        let result: HashMap<String, String> = coerce_and_validate(value).unwrap();
        assert_eq!(result.get("metadata"), Some(&"{\"key\":\"value\"}".to_string()));
    }

    #[test]
    fn test_coerce_array_to_json_string() {
        let value = json!({"tags": ["tag1", "tag2"]});
        let result: HashMap<String, String> = coerce_and_validate(value).unwrap();
        assert_eq!(result.get("tags"), Some(&"[\"tag1\",\"tag2\"]".to_string()));
    }

    #[test]
    fn test_fast_path_string_passes_through() {
        let value = json!({"name": "test"});
        let result: HashMap<String, String> = coerce_and_validate(value).unwrap();
        assert_eq!(result.get("name"), Some(&"test".to_string()));
    }

    #[test]
    fn test_coercion_error_with_path() {
        let error = CoercionError::new(
            "Test error",
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid"),
            Some(json!(123)),
        )
        .with_parameter_path("session_id".to_string())
        .with_expected_type("UUID string")
        .with_hint("Create a session first");

        assert_eq!(error.parameter_path, Some("session_id".to_string()));
        assert_eq!(error.expected_type, Some("UUID string".to_string()));
        assert_eq!(error.hint, Some("Create a session first".to_string()));
    }

    #[test]
    fn test_to_mcp_error_format() {
        let error = CoercionError::new(
            "Invalid parameter",
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "test"),
            Some(json!(123)),
        )
        .with_parameter_path("session_id".to_string())
        .with_expected_type("UUID string")
        .with_hint("Use session tool to create");

        let mcp_error = error.to_mcp_error();
        // The to_mcp_error() function serializes the error details into the message
        // The second parameter to invalid_params (data) is None in our implementation
        // So we verify the message contains our structured data
        let error_message = mcp_error.message;
        assert!(error_message.contains("session_id"));

        // Parse the message as JSON to verify structure
        let error_data: serde_json::Value = serde_json::from_str(&error_message).unwrap();
        assert_eq!(error_data["parameter"], "session_id");
        assert_eq!(error_data["expectedType"], "UUID string");
        assert_eq!(error_data["receivedValue"], 123);
        assert_eq!(error_data["hint"], "Use session tool to create");
    }
}
