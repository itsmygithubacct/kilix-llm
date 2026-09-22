import unittest

from kilix_llm.supervised import encode_target, target_positions, target_loss


class Tokenizer:
    eos_token_id = 9
    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return [1, 2, 3]
    def encode(self, answer, **options):
        assert options == dict(add_special_tokens=False)
        return [7, 8]


class SupervisedTests(unittest.TestCase):
    def test_assistant_mask_and_causal_alignment(self):
        row = encode_target(Tokenizer(), [{"role":"user","content":"test"}], "answer", 6)
        self.assertEqual(row, {"input_ids":[1,2,3,7,8,9],"labels":[-100,-100,-100,7,8,9]})
        self.assertEqual(target_positions(row["labels"]), ([2,3,4],[7,8,9]))

    def test_no_silent_truncation_or_empty_target(self):
        with self.assertRaises(ValueError):encode_target(Tokenizer(), [], "answer", 5)
        with self.assertRaises(ValueError):encode_target(Tokenizer(), [], " ", 8)
        for labels in ([-100,-100], [7,8]):
            with self.assertRaises(ValueError):target_positions(labels)

    def test_selective_logits_match_full_loss_and_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional torch runtime not installed')
        from types import SimpleNamespace
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding=torch.nn.Embedding(10,4)
                self.head=torch.nn.Linear(4,10)
            def forward(self,input_ids,attention_mask,use_cache,logits_to_keep):
                hidden=self.embedding(input_ids)
                return SimpleNamespace(logits=self.head(hidden[:,logits_to_keep,:]))
        torch.manual_seed(7)
        model=Model()
        row={'input_ids':[1,2,3,7,8,9],'labels':[-100,-100,-100,7,8,9]}
        selected=target_loss(model,row,'cpu')
        selected.backward()
        selected_grads=[p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        logits=model.head(model.embedding(torch.tensor([row['input_ids']])))
        full=torch.nn.functional.cross_entropy(logits[0,:-1],torch.tensor(row['labels'][1:]),ignore_index=-100)
        full.backward()
        torch.testing.assert_close(selected,full)
        for actual,expected in zip((p.grad for p in model.parameters()),selected_grads):
            torch.testing.assert_close(actual,expected)
