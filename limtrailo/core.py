# -*- coding: utf-8 -*-
from . import helpers

def get_hmm():
    """Get a thought."""
    return 'hmmm...'


def hmm():
    """Contemplation..."""
    if helpers.get_answer():
        print(get_hmm())

def end_to_end(train_data, test_data, d, model_class, hparams):
  train_data = torch.utils.data.DataLoader(train_data,
                                           batch_size=hparams['BATCH_SIZE'],
                                           shuffle=True)
  test_data = torch.utils.data.DataLoader(test_data,
                                          batch_size=hparams['BATCH_SIZE'])
  
  net = model_class(d=d)
  optimizer = torch.optim.Adam(net.parameters(),
                              lr=hparams['LR'])
  
  test_scores = []
  train_scores = []
  
  def get_test_metrics():
    test_neglogp = np.array(0.)
    samples = 0

    for x in test_data:

      with torch.no_grad():
        test_neglogp += np.sum(net.get_neglogp(x).cpu().numpy())

      samples += x.shape[0]

    return test_neglogp / samples

  test_scores.append(get_test_metrics())
  
  for epoch in range(hparams['EPOCHS']):

    for x in train_data:
      optimizer.zero_grad()

      neglogp = net.get_neglogp(x)
      loss = torch.sum(neglogp)

      loss.backward()
      optimizer.step()

      train_scores.append(np.sum(neglogp.detach().cpu().numpy()) / x.shape[0])

    test_scores.append(get_test_metrics())

  with torch.no_grad():
    return (train_scores,
            test_scores,
            net.get_distribution().cpu().numpy())
  return 