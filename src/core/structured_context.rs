// Copyright (c) 2025 Julius ML
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
use chrono::{DateTime, Utc};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StructuredContext {
    pub key_decisions: Vec<DecisionItem>,
    pub open_questions: Vec<QuestionItem>,
    pub key_concepts: Vec<ConceptItem>,
    pub technical_specifications: Vec<SpecItem>,
    pub action_items: Vec<ActionItem>,
    pub references: Vec<ReferenceItem>,
    pub conversation_flow: Vec<FlowItem>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DecisionItem {
    pub description: String,
    pub context: String,
    pub alternatives: Vec<String>,
    pub confidence: f32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QuestionItem {
    pub question: String,
    pub context: String,
    pub status: QuestionStatus,
    pub timestamp: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum QuestionStatus {
    Open,
    InProgress,
    Answered,
    Deferred,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConceptItem {
    pub name: String,
    pub definition: String,
    pub examples: Vec<String>,
    pub related_concepts: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpecItem {
    pub title: String,
    pub description: String,
    pub requirements: Vec<String>,
    pub constraints: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ActionItem {
    pub description: String,
    pub assigned_to: Option<String>,
    pub due_date: Option<DateTime<Utc>>,
    pub status: ActionStatus,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ActionStatus {
    Todo,
    InProgress,
    Completed,
    Blocked,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReferenceItem {
    pub title: String,
    pub url: Option<String>,
    pub file_path: Option<String>,
    pub description: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlowItem {
    pub step_description: String,
    pub timestamp: DateTime<Utc>,
    pub related_updates: Vec<uuid::Uuid>,
    pub outcome: Option<String>,
}

impl Default for StructuredContext {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuredContext {
    pub fn new() -> Self {
        Self {
            key_decisions: Vec::new(),
            open_questions: Vec::new(),
            key_concepts: Vec::new(),
            technical_specifications: Vec::new(),
            action_items: Vec::new(),
            references: Vec::new(),
            conversation_flow: Vec::new(),
        }
    }
}
