//
// Created by Jonas on 19/05/2022.
//

#ifndef MESHCORE_RENDERMODELCONTROLWIDGET_H
#define MESHCORE_RENDERMODELCONTROLWIDGET_H

#include <QWidget>
#include "AbstractRenderModel.h"

namespace Ui {
    class RenderModelControlWidget;
}

class RenderModelControlWidget: public QWidget
{
    Q_OBJECT
public:
    explicit RenderModelControlWidget(const std::shared_ptr<AbstractRenderModel>& renderModel, QWidget* parent);
    ~RenderModelControlWidget() override;

private:
    Ui::RenderModelControlWidget *ui;
    std::shared_ptr<AbstractRenderModel> renderModel;
    std::shared_ptr<SimpleRenderModelListener> listener;
};


#endif //MESHCORE_RENDERMODELCONTROLWIDGET_H
