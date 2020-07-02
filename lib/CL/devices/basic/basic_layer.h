#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_context.h"
#include "pocl_workgroup_func.h"



struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;

  /* List of commands ready to be executed */
  _cl_command_node * volatile ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node * volatile command_list;
  /* Lock for command list related operations */
  pocl_lock_t cq_lock;
  /* printf buffer */
  void *printf_buffer;
};

void
montage_riscv_device_run (void *data, _cl_command_node *cmd);
void
montage_riscv_gencode(_cl_command_node *command);