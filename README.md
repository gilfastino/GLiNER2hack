# GLiNER2: Unified Schema-Based Information Extraction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/gliner2.svg)](https://badge.fury.io/py/gliner2)
[![Downloads](https://pepy.tech/badge/gliner2)](https://pepy.tech/project/gliner2)

> *Next-generation information extraction for entities, structures, and classification—unified in one efficient model.*

GLiNER2 is a state-of-the-art information extraction library that unifies **Named Entity Recognition (NER)**, **Text Classification**, and **Hierarchical Structure Extraction** into a single, efficient model. Built on transformer architecture, it delivers competitive performance while maintaining CPU efficiency and compact size (205M parameters).

## 🚀 Key Features

- **🎯 Unified Extraction**: Entities, classification, and structured data in one model
- **💻 CPU Efficient**: Fast inference on standard hardware without GPU requirements  
- **📊 Schema-Driven**: Intuitive API for defining complex extraction tasks
- **🔒 Privacy-First**: Local deployment with no external API dependencies
- **⚡ Multi-Task**: Combine multiple extraction tasks in a single forward pass
- **🎨 Flexible**: Support for descriptions, and field types

## 📦 Installation

```bash
pip install gliner2
```

## 🔥 Quick Start

```python
from gliner2 import GLiNER2

# Load the model
extractor = GLiNER2.from_pretrained("fastino/gliner2-base")

# Extract entities
text = "Apple CEO Tim Cook announced the iPhone 15 in Cupertino."
entities = extractor.extract_entities(text, ["company", "person", "product", "location"])

print(entities)
# Output: {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}
```

## 🎯 Core Capabilities

### 1. Named Entity Recognition

Extract entities with natural language descriptions for better accuracy:

```python
# Simple entity extraction
entities = extractor.extract_entities(
    "Dr. Sarah Johnson from Stanford published groundbreaking AI research.",
    ["person", "organization", "field"]
)
# Output: {'entities': {'person': ['Dr. Sarah Johnson'], 'organization': ['Stanford'], 'field': ['AI research']}}

# With descriptions for better accuracy
entities = extractor.extract_entities(
    "Patient took 400mg ibuprofen for headache.",
    {
        "medication": "Names of drugs and medications",
        "dosage": "Dosage amounts like '400mg' or '2 tablets'",
        "symptom": "Medical symptoms or conditions"
    }
)
# Output: {'entities': {'medication': ['ibuprofen'], 'dosage': ['400mg'], 'symptom': ['headache']}}
```

### 2. Text Classification

Single-label and multi-label classification with configurable thresholds:

```python
# Simple sentiment classification
result = extractor.classify_text(
    "This product is amazing! Best purchase ever.",
    {"sentiment": ["positive", "negative", "neutral"]}
)
# Output: {'sentiment': 'positive'}

# Multi-label classification with custom threshold
result = extractor.classify_text(
    "Great camera but poor battery life.",
    {
        "aspects": {
            "labels": ["camera", "battery", "display", "performance"],
            "multi_label": True,
            "cls_threshold": 0.4
        }
    }
)
# Output: {'aspects': ['camera', 'battery']}
```

### 3. Structured Data Extraction

Extract complex hierarchical structures with field-level control:

```python
# Extract structured product information
text = "iPhone 15 Pro costs $999 with 128GB storage, 5G connectivity."

results = extractor.extract_json(
    text,
    {
        "product": [
            "name::str::Product name and model",
            "price::str::Product cost", 
            "storage::str::Storage capacity",
            "features::list::Key product features"
        ]
    }
)
# Output: {'product': [{'name': 'iPhone 15 Pro', 'price': '$999', 'storage': '128GB', 'features': ['5G connectivity']}]}

# Multiple structures in one pass
results = extractor.extract_json(
    "Apple Inc. headquarters in Cupertino launched iPhone 15 for $999.",
    {
        "company": [
            "name::str::Company name",
            "location::str::Company location"
        ],
        "product": [
            "name::str::Product name",
            "price::str::Product price"
        ]
    }
)
# Output: {'company': [{'name': 'Apple Inc.', 'location': 'Cupertino'}], 'product': [{'name': 'iPhone 15', 'price': '$999'}]}
```

### 4. Multi-Task Schema Composition

Combine all extraction types in a single, efficient inference:

```python
# Comprehensive extraction schema
schema = (extractor.create_schema()
    # Extract entities
    .entities(["person", "company", "product", "location"])
    
    # Classify sentiment and urgency
    .classification("sentiment", ["positive", "negative", "neutral"])
    .classification("urgency", ["low", "medium", "high"])
    
    # Extract structured product info
    .structure("product_info")
        .field("name", dtype="str")
        .field("price", dtype="str") 
        .field("features", dtype="list")
        .field("category", dtype="str", choices=["electronics", "software", "service"])
)

# Extract everything in one pass
text = "Apple CEO Tim Cook announced iPhone 15 for $999. This is exciting news!"
results = extractor.extract(text, schema)
# Output: {
#     'entities': {'person': ['Tim Cook'], 'company': ['Apple'], 'product': ['iPhone 15'], 'location': []},
#     'sentiment': 'positive',
#     'urgency': 'medium',
#     'product_info': [{'name': 'iPhone 15', 'price': '$999', 'features': [], 'category': 'electronics'}]
# }
```

## 🎨 Advanced Features

### Custom Thresholds and Precision Control

```python
schema = (extractor.create_schema()
    .structure("financial_transaction")
        .field("amount", dtype="str", threshold=0.95)      # High precision for money
        .field("date", dtype="str", threshold=0.8)         # Medium precision for dates
        .field("description", dtype="str", threshold=0.6)  # Lower precision for descriptions
)
```

### Field Types and Constraints

```python
schema = (extractor.create_schema()
    .structure("product_review")
        .field("product", dtype="str")                     # Single value
        .field("features", dtype="list")                   # Multiple values
        .field("rating", dtype="str", choices=["1", "2", "3", "4", "5"])  # Constrained choices
        .field("tags", dtype="list", choices=["recommended", "budget", "premium"])  # Multi-select
)
```

## 🏭 Real-World Applications

### Healthcare Information Extraction

```python
medical_text = "Patient John Smith, 65, visited Dr. Roberts on March 15th for chest pain. Prescribed 50mg aspirin daily."

schema = (extractor.create_schema()
    .entities({
        "patient": "Patient names and identifiers",
        "doctor": "Healthcare provider names", 
        "medication": "Prescribed drugs and medications",
        "symptom": "Medical symptoms and conditions"
    })
    .classification("urgency", ["low", "medium", "high", "critical"])
    .structure("prescription")
        .field("medication", dtype="str")
        .field("dosage", dtype="str")
        .field("frequency", dtype="str")
)

results = extractor.extract(medical_text, schema)
# Output: {
#     'entities': {'patient': ['John Smith'], 'doctor': ['Dr. Roberts'], 'medication': ['aspirin'], 'symptom': ['chest pain']},
#     'urgency': 'medium',
#     'prescription': [{'medication': 'aspirin', 'dosage': '50mg', 'frequency': 'daily'}]
# }
```

### Legal Document Processing

```python
legal_text = "Employment Agreement between TechCorp Inc. and Jane Doe, effective January 1, 2024..."

schema = (extractor.create_schema()
    .entities(["company", "person", "date", "contract_type"])
    .classification("document_type", ["employment", "service", "nda", "partnership"])
    .structure("contract_details")
        .field("parties", dtype="list")
        .field("effective_date", dtype="str")
        .field("termination_clause", dtype="str")
        .field("compensation", dtype="str")
)

results = extractor.extract(legal_text, schema)
# Output: {
#     'entities': {'company': ['TechCorp Inc.'], 'person': ['Jane Doe'], 'date': ['January 1, 2024'], 'contract_type': ['Employment Agreement']},
#     'document_type': 'employment',
#     'contract_details': [{'parties': ['TechCorp Inc.', 'Jane Doe'], 'effective_date': 'January 1, 2024', 'termination_clause': '', 'compensation': ''}]
# }
```

### E-commerce and Product Analysis

```python
product_text = "MacBook Pro 16-inch with M3 chip, 32GB RAM, 1TB SSD. Price: $3,499. Great for video editing!"

schema = (extractor.create_schema()
    .entities(["product", "brand", "price", "specification"])
    .classification("category", ["laptop", "desktop", "tablet", "phone"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .structure("product_specs")
        .field("name", dtype="str")
        .field("processor", dtype="str")
        .field("memory", dtype="str")
        .field("storage", dtype="str")
        .field("use_cases", dtype="list")
)

results = extractor.extract(product_text, schema)
# Output: {
#     'entities': {'product': ['MacBook Pro 16-inch'], 'brand': ['MacBook'], 'price': ['$3,499'], 'specification': ['M3 chip', '32GB RAM', '1TB SSD']},
#     'category': 'laptop',
#     'sentiment': 'positive',
#     'product_specs': [{'name': 'MacBook Pro 16-inch', 'processor': 'M3 chip', 'memory': '32GB RAM', 'storage': '1TB SSD', 'use_cases': ['video editing']}]
# }
```

## 🔧 API Reference

### Core Methods

| Method | Description |
|--------|-------------|
| `GLiNER2.from_pretrained(model_name)` | Load a pre-trained model |
| `create_schema()` | Create a new extraction schema |
| `extract(text, schema, **kwargs)` | Main extraction method |
| `extract_entities(text, entities, **kwargs)` | Quick entity extraction |
| `classify_text(text, tasks, **kwargs)` | Quick text classification |
| `extract_json(text, structures, **kwargs)` | Quick structured extraction |

## 📊 Performance

GLiNER2 achieves competitive performance across diverse tasks while maintaining efficiency:

| Task | Dataset | GLiNER2 | GPT-4o | Baseline |
|------|---------|---------|---------|----------|
| NER | CrossNER | 0.590 | 0.589 | 0.615 |
| Classification | SNIPS | 0.83 | 0.97 | 0.77 |
| Classification | Banking77 | 0.70 | 0.78 | 0.42 |
| Classification | SST-2 | 0.86 | 0.94 | 0.92 |

**Speed Comparison**: 3.25× faster than GPT-4o while running on CPU

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use GLiNER2 in your research, please cite:

```bibtex
@article{zaratiana2024gliner2,
  title={GLiNER2: Schema-Driven Multi-Task Learning for Structured Information Extraction},
  author={Zaratiana, Urchade and Pasternak, Gil and Boyd, Oliver and Hurn-Maloney, George and Lewis, Ash},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 🙏 Acknowledgments

- Built upon the original [GLiNER](https://github.com/urchade/GLiNER) architecture
- Developed by the team at [Fastino AI](https://fastino.ai)
- Thanks to the open-source community for feedback and contributions
---

<div align="center">
    <strong>Built with ❤️ by the Fastino AI team</strong><br>
    <em>Making advanced information extraction accessible to everyone</em>
</div>